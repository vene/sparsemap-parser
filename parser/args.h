#pragma once
#include <iostream>

struct ParserArgs
{
    bool cost_augment = true;
    bool sparsemap = false;
    bool marginal = false;
    bool print_active_set_sizes = false;
    bool print_valid_predictions = false;
    unsigned max_iter = 20;
    unsigned max_decode_iter = 100;
    unsigned batch_size = 16;
    unsigned hidden_dim = 100;
    unsigned pos_dim = 25;
    unsigned rnn_dim = 125;
    unsigned rnn_layers=1;
    float word_dropout = 0.25;
    float init_lr = 0.001;
    float decay = 0.9;
    std::string scorer = "mlp";
    std::string lang;
    std::string saved_model;

    void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc)
        {
            std::string arg = argv[i];
            if (arg == "--no-cost-augment")
            {
                cost_augment = false;
                i += 1;
            }
            else if (arg == "--sparsemap")
            {
                sparsemap = true;
                i += 1;
            }
            else if (arg == "--marginal")
            {
                marginal = true;
                i += 1;
            }
            else if (arg == "--lang")
            {
                assert(i + 1 < argc);
                lang = argv[i + 1];
                i += 2;
            }
            else if (arg == "--scorer")
            {
                assert(i + 1 < argc);
                scorer = argv[i + 1];
                i += 2;
            }
            else if (arg == "--max-iter")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> max_iter;
                i += 2;
            }
            else if (arg == "--rnn-layers")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> rnn_layers;
                i += 2;
            }
            else if (arg == "--max-decode-iter")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> max_decode_iter;
                i += 2;
            }
            else if (arg == "--hidden-dim")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> hidden_dim;
                i += 2;
            }
            else if (arg == "--pos-dim")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> pos_dim;
                i += 2;
            }
            else if (arg == "--word-dropout")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> word_dropout;
                i += 2;
            }
            else if (arg == "--rnn-dim")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> rnn_dim;
                i += 2;
            }
            else if (arg == "--batch-size")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> batch_size;
                i += 2;
            }
            else if (arg == "--init-lr")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> init_lr;
                i += 2;
            }
            else if (arg == "--decay")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> decay;
                i += 2;
            }
            else if (arg == "--print-active-set-sizes")
            {
                print_active_set_sizes = true;
                i += 1;
            }
            else if (arg == "--print-valid-predictions")
            {
                print_valid_predictions = true;
                i += 1;
            }
            else
            {
                i += 1;
            }
        }

        if (sparsemap && marginal)
        {
            std::cerr << "cannot use sparsemap and marginal together" << std::endl;
            throw 1;
        }
    }

};


std::ostream& operator << (std::ostream &o, const ParserArgs &args)
{
    o << "Arguments:" << std::endl
      << "\n    Language: " << args.lang
      << "\n   Sparsemap: " << args.sparsemap
      << "\n    Marginal: " << args.marginal
      << "\nCost augment: " << args.cost_augment
      << "\n  Batch size: " << args.batch_size
      << "\n  Hidden dim: " << args.hidden_dim
      << "\n     POS dim: " << args.pos_dim
      << "\n     RNN dim: " << args.rnn_dim
      << "\n Scorer type: " << args.scorer
      << "\n  RNN layers: " << args.rnn_layers
      << "\nWord dropout: " << args.word_dropout
      << "\n   Max. iter: " << args.max_iter
      << "\n Max. decode: " << args.max_decode_iter
      << "\n  Initial lr: " << args.init_lr
      << "\n  Decay rate: " << args.decay
      << std::endl;
    return o;
}
