# pragma once
#include <dynet/expr.h>
#include <dynet/rnn.h>
#include <dynet/lstm.h>
#include "dynet/io.h"

#include "scorers.h"
#include "mst_losses.h"

namespace dy = dynet;

using dy::Expression;
using dy::Parameter;
using dy::LookupParameter;
using dy::parameter;

using std::vector;
using std::unique_ptr;

enum Special { UNK=0, BOS, EOS };


void incremental_forward_all(dy::ComputationGraph& cg,
                             vector<Expression> exprs)
{
    Expression* max_expr = &(exprs[0]);
    for (auto && expr : exprs)
        if (expr.i > max_expr->i)
            max_expr = &expr;
    cg.incremental_forward(*max_expr);
}

struct Parser
{
    std::vector<float> vocab_;
    std::vector<float> posvocab_;
    unsigned hidden_dim_;
    unsigned pos_dim_;
    unsigned rnn_dim_;
    unsigned rnn_layers_;
    float word_dropout_;
    bool cost_augment_;
    bool sparsemap_;
    bool marginal_;
    unsigned decode_iter_;
    dy::ParameterCollection p_;
    unique_ptr<dy::RNNBuilder> fw_builder, bw_builder,
                               fw2_builder, bw2_builder;
    unique_ptr<EdgeScorer> scorer;

    dy::LookupParameter p_emb, p_pos;

    explicit Parser(dy::ParameterCollection& params,
                    std::vector<float> vocab,
                    std::vector<float> posvocab,
                    unsigned hidden_dim,
                    unsigned pos_dim,
                    unsigned rnn_dim,
                    unsigned rnn_layers,
                    float word_dropout,
                    bool cost_augment=true,
                    bool sparsemap=false,
                    bool marginal=false,
                    unsigned decode_iter=100,
                    std::string scorer_type="mlp")
        : vocab_(vocab)
        , posvocab_(posvocab)
        , hidden_dim_(hidden_dim)
        , pos_dim_(pos_dim)
        , rnn_dim_(rnn_dim)
        , rnn_layers_(rnn_layers)
        , word_dropout_(word_dropout)
        , cost_augment_(cost_augment)
        , sparsemap_(sparsemap)
        , marginal_(marginal)
        , decode_iter_(decode_iter)
    {
        p_ = params.add_subcollection("parser");
        p_emb = p_.add_lookup_parameters(vocab_.size(), {hidden_dim});
        p_pos = p_.add_lookup_parameters(posvocab_.size(), {pos_dim});

        // bidirectiona lstm
        const unsigned layers = 1;
        unsigned input_dim = hidden_dim + pos_dim;
        fw_builder.reset(
            new dy::VanillaLSTMBuilder(layers, input_dim, rnn_dim, p_));
        bw_builder.reset(
            new dy::VanillaLSTMBuilder(layers, input_dim, rnn_dim, p_));

        if (rnn_layers == 2)
        {
            fw2_builder.reset(
                new dy::VanillaLSTMBuilder(layers, 2 * rnn_dim, rnn_dim, p_));
            bw2_builder.reset(
                new dy::VanillaLSTMBuilder(layers, 2 * rnn_dim, rnn_dim, p_));
        }
        else if (rnn_layers > 2 || rnn_layers < 0)
            throw std::invalid_argument("rnn_layers must be 1 or 2");

        // arc scorer
        if (scorer_type == "mlp")
            scorer.reset(new MLPScorer(p_, hidden_dim, 2 * rnn_dim));
        else if (scorer_type == "bilinear")
            scorer.reset(new BilinearScorer(p_, hidden_dim, 2 * rnn_dim));
        else
            throw std::invalid_argument("Invalid scorer, need mlp|bilinear");
    }

    void save(const std::string filename)
    {
        dy::TextFileSaver s(filename);
        s.save(p_);
    }

    void load(const std::string filename)
    {
        dy::TextFileLoader l(filename);
        l.populate(p_);
    }

    vector<Expression> batch_potentials(dy::ComputationGraph& cg,
                                        const vector<Sentence> batch,
                                        bool training)
    {
        fw_builder->new_graph(cg);
        bw_builder->new_graph(cg);
        if (rnn_layers_ == 2)
        {
            fw2_builder->new_graph(cg);
            bw2_builder->new_graph(cg);
        }
        scorer->new_graph(cg);

        vector<Expression> potentials;
        vector<Expression> tmp;

        std::uniform_real_distribution<float> unif(0, 1);

        for (auto&& sent : batch)
        {
            // embed sentence;
            vector<Expression> emb;

            emb.push_back(dy::concatenate({
                dy::lookup(cg, p_emb, Special::BOS),
                dy::lookup(cg, p_pos, Special::BOS)
            }));

            for (int i = 0; i < sent.tok_ixs.size(); ++i)
            {
                unsigned word_ix = sent.tok_ixs[i];
                unsigned pos_ix = sent.pos_ixs[i];

                // perform word dropout:
                if (training)
                {
                    float count = vocab_[word_ix];

                    if (count <= 0)
                     std::cerr << "this shouldn't happen " << word_ix << std::endl;

                    float p = count;
                    p /= (p + word_dropout_);
                    float u = unif(*rndeng);
                    // std::cout << p << " " << u << std::endl;

                    if (u >= p) // drop word
                    {
                        // std::cout << "Dropping word with count " << count << std::endl;
                        word_ix = Special::UNK;
                    }
                }

                emb.push_back(dy::concatenate({
                    dy::lookup(cg, p_emb, word_ix),
                    dy::lookup(cg, p_pos, pos_ix)
                }));
            }

            emb.push_back(dy::concatenate({
                dy::lookup(cg, p_emb, Special::EOS),
                dy::lookup(cg, p_pos, Special::EOS)
            }));

            // run rnns in both directions
            fw_builder->start_new_sequence();
            bw_builder->start_new_sequence();

            size_t n_toks = sent.tok_ixs.size();
            size_t n_toks_padded = emb.size();

            vector<Expression> enc_fw(n_toks_padded);
            vector<Expression> enc_bw(n_toks_padded);
            vector<Expression> enc(n_toks);

            for (size_t i = 0; i < n_toks_padded; ++i)
            {
                size_t j = n_toks_padded - i - 1;
                enc_fw[i] = fw_builder->add_input(emb[i]);
                enc_bw[j] = bw_builder->add_input(emb[j]);
            }

            for (size_t i = 0; i < n_toks; ++i)
                enc[i] = dy::concatenate({enc_fw[i + 1], enc_bw[i + 1]});

            // if deep lstm
            if (rnn_layers_ == 2)
            {
                fw2_builder->start_new_sequence();
                bw2_builder->start_new_sequence();
                for (size_t i = 0; i < n_toks; ++i)
                {
                    size_t j = n_toks - i - 1;
                    enc_fw[i] = fw2_builder->add_input(enc[i]);
                    enc_bw[j] = bw2_builder->add_input(enc[j]);
                }

                for (size_t i = 0; i < n_toks; ++i)
                    enc[i] = dy::concatenate({enc_fw[i], enc_bw[i]});
            }

            // make potentials
            auto S = scorer->make_potentials(enc);
            potentials.push_back(S);
        }
        return potentials;
    }

    void gather_predictions(dy::ComputationGraph& cg,
                            const vector<Sentence> batch,
                            vector<SparsemapSolution>& solutions)
    {
        auto potentials = batch_potentials(cg, batch, false);
        incremental_forward_all(cg, potentials);
        for (auto && theta : potentials)
            solutions.push_back(sparsemap_decode(theta.value(), decode_iter_));
    }


    unsigned count_correct(dy::ComputationGraph& cg,
                           const vector<Sentence> batch,
                           bool print_active_set_sizes)
    {
        unsigned correct = 0;
        auto potentials = batch_potentials(cg, batch, false);
        incremental_forward_all(cg, potentials);
        for (int i = 0; i < batch.size(); ++i)
        {
            if (print_active_set_sizes && sparsemap_) // gather statistics
            {
                Expression loss;
                if (cost_augment_)
                    loss = sparsemap_hinge_loss_tree(potentials[i],
                                                     batch[i].heads.size(),
                                                     batch[i].heads,
                                                     decode_iter_,
                                                     true);
                else
                    loss = sparsemap_loss_tree(potentials[i],
                                               batch[i].heads.size(),
                                               batch[i].heads,
                                               decode_iter_,
                                               true);
                cg.incremental_forward(loss);
            }

            auto heads_pred = mst_decode(potentials[i].value());

            for (int m = 1; m < heads_pred.size(); ++m)
                if (heads_pred[m] == batch[i].heads[m])
                    correct += 1;
        }
        return correct;

    }

    Expression batch_loss(dy::ComputationGraph& cg,
                          const vector<Sentence> batch)
    {
        auto potentials = batch_potentials(cg, batch, true);

        vector<Expression> objs;
        for (int i = 0; i < batch.size(); ++i)
        {
            auto y_true = batch[i].heads;
            auto length = y_true.size();

            Expression loss;
            if (sparsemap_)
            {
                if (cost_augment_)
                    loss = sparsemap_hinge_loss_tree(potentials[i], length, y_true, decode_iter_);
                else
                    loss = sparsemap_loss_tree(potentials[i], length, y_true, decode_iter_);
            }
            else if (marginal_)
            {
                if (cost_augment_)
                    throw 1;
                else
                    loss = marginal_loss_tree(potentials[i], length, y_true);
            }
            else
            {
                if (cost_augment_)
                    loss = struct_hinge_loss_tree(potentials[i], length, y_true);
                else
                    loss = perceptron_loss_tree(potentials[i], length, y_true);
            }
            objs.push_back(loss);
        }

        return dy::sum(objs);
    }
};
