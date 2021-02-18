// Parse with gold segmentation
#include <fstream>
#include <chrono>

#include <dynet/dynet.h>
#include <dynet/globals.h>
#include <dynet/training.h>
#include <dynet/timing.h>

#include "utils.h"
#include "args.h"
#include "parser.h"

namespace dy = dynet;

using std::cout;
using std::endl;


size_t count_words(vector<vector<Sentence> >& dataset)
{
    size_t n_words = 0;
    for (auto&& batch : dataset)
        for (auto&& sent : batch)
            n_words += sent.tok_ixs.size();
    return n_words;
}


int main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);
    dy::initialize(dyparams);

    ParserArgs args;
    args.parse(argc, argv);
    cout << args << endl;

    std::stringstream vocab_fn, posvocab_fn, train_fn, valid_fn, test_fn;
    vocab_fn << "data/" << args.lang << "-vocab.txt";
    posvocab_fn << "data/" << args.lang << "-posvocab.txt";
    train_fn << "data/" << args.lang << "-train.txt";
    valid_fn << "data/" << args.lang << "-valid.txt";
    test_fn  << "data/" << args.lang << "-test-" << args.lang << ".txt";

    auto vocab = read_vocab(vocab_fn.str());
    auto posvocab = read_vocab(posvocab_fn.str());

    auto train_batches = read_batches(train_fn.str(), args.batch_size);
    auto valid_batches = read_batches(valid_fn.str(), args.batch_size);
    auto test_batches = read_batches(test_fn.str(), args.batch_size);
    auto n_batches = train_batches.size();

    unsigned n_train_sents = 0;
    for (auto&& batch : train_batches)
        n_train_sents += batch.size();
    cout << "Training on " << n_train_sents << " sentences." << endl;

    auto n_train_words = count_words(train_batches);
    auto n_valid_words = count_words(test_batches);
    auto n_test_words = count_words(test_batches);

    vector<vector<vector<Sentence> >::iterator> train_iter(n_batches);
    std::iota(train_iter.begin(), train_iter.end(), train_batches.begin());

    dy::ParameterCollection params;
    std::unique_ptr<Parser> parser(new Parser(params,
                                              vocab,
                                              posvocab,
                                              args.hidden_dim,
                                              args.pos_dim,
                                              args.rnn_dim,
                                              args.rnn_layers,
                                              args.word_dropout,
                                              args.cost_augment,
                                              args.sparsemap,
                                              args.marginal,
                                              args.max_decode_iter,
                                              args.scorer));

    // dy::SimpleSGDTrainer trainer(params, args.init_lr);
    dy::AdamTrainer trainer(params, args.init_lr);
    std::chrono::duration<double> elapsed;
    float best_valid_uas = 0;

    for (unsigned it = 0; it < args.max_iter; ++it)
    {
        // shuffle the permutation vector
        std::shuffle(train_iter.begin(), train_iter.end(), *dy::rndeng);

        float total_loss = 0;

        // train an epoch
        auto tic = std::chrono::system_clock::now();
        for (auto&& batch : train_iter)
        {
            dy::ComputationGraph cg;
            // std::cout << "Sentence size " << (*batch)[0].heads.size() << std::endl;
            auto loss = parser->batch_loss(cg, *batch);
            auto loss_val = dy::as_scalar(cg.incremental_forward(loss));
            total_loss += loss_val;
            cg.backward(loss);
            trainer.update();
        }
        auto toc = std::chrono::system_clock::now();
        elapsed += toc - tic;
        cout << "Epoch done. Validating" << endl;

        // compute training scores
        unsigned train_correct = 0;
        for (auto&& batch : train_batches)
        {
            dy::ComputationGraph cg;
            auto count = parser->count_correct(cg, batch, args.print_active_set_sizes);
            train_correct += count;
        }
        if (args.print_active_set_sizes)
            std::cerr << "\t";

        float train_uas = float(train_correct) / n_train_words;

        // compute validation scores
        unsigned valid_correct = 0;
        for (auto&& batch : valid_batches)
        {
            dy::ComputationGraph cg;
            auto count = parser->count_correct(cg, batch, args.print_active_set_sizes);
            valid_correct += count;
        }
        if (args.print_active_set_sizes)
            std::cerr << "\t";

        float valid_uas = float(valid_correct) / n_valid_words;

        unsigned test_correct = 0;
        for (auto&& batch : test_batches)
        {
            dy::ComputationGraph cg;
            auto count = parser->count_correct(cg, batch, args.print_active_set_sizes);
            test_correct += count;
        }
        if (args.print_active_set_sizes)
            std::cerr << "\t";
        float test_uas = float(test_correct) / n_test_words;

        cout << "epoch " << it << '\n'
             << " train loss " << total_loss / n_train_sents << '\n'
             << "  train UAS " << train_uas << '\n'
             << "  valid UAS " << valid_uas << '\n'
             << "   test UAS " << test_uas << '\n'
             << "    elapsed " << elapsed.count() << endl;

        if (valid_uas < best_valid_uas)
        {
            // cout << "decaying learning rate" << endl;
            // trainer.learning_rate *= args.decay;
        }
        else
        {
            best_valid_uas = valid_uas;
        }

        if (total_loss == 0)
        {
            cout << "Converged." << endl;
            break;
        }
    }
    cout << "Peak validation UAS\n" << best_valid_uas << endl;

    if (args.print_valid_predictions)
    {
        vector<SparsemapSolution> solutions;
        for (auto&& batch : valid_batches)
        {
            dy::ComputationGraph cg;
            parser->gather_predictions(cg, batch, solutions);
        }
        for (auto&& sol : solutions)
            std::cerr << sol << std::endl;
    }

    return 0;
}
