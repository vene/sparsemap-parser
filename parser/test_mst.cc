#include <random>
#include <cassert>
#include <dynet/expr.h>
#include <dynet/tensor.h>
#include <dynet/tensor-eigen.h>
#include <dynet/grad-check.h>

#include "mst_losses.h"

using std::cout;
using std::endl;
using AD3::Arc;

namespace dy = dynet;

void test_equivalent(const unsigned int seed, const unsigned int n_words)
{
    random_device rd;
    mt19937 gen(rd());
    gen.seed(seed);
    normal_distribution<> nrm;

    // create arcs as in reference example
    vector<Arc*> arcs;
    for (int m = 1; m < n_words; ++m)
    {
        for (int h = 0; h < n_words; ++h)
        {
            if (h == m) continue;
            Arc *arc = new Arc(h, m);
            arcs.push_back(arc);
        }
    }

    vector<double> arc_scores(arcs.size(), 0);

    dy::ComputationGraph cg;
    dy::Expression arc_phi_dy = dy::zeros(cg, {n_words, n_words - 1});
    dy::Tensor arc_phi = arc_phi_dy.value();

    size_t k = 0;
    for (auto&& arc : arcs)
    {
        double val = nrm(gen);
        arc_scores[k] = val;
        mat(arc_phi)(arc->head(), arc->modifier() - 1) = val;
        k += 1;
    }

    // using FactorTreeFast
    auto fast_mst = mst_decode(arc_phi);

    // using the usual factor
    AD3::FactorTree tree_factor;
    tree_factor.Initialize(n_words, arcs);

    vector<int> standard_mst;
    double val;
    tree_factor.RunCLE(arc_scores, &standard_mst, &val);

    for (int k = 0; k < standard_mst.size(); k++)
        assert(standard_mst[k] == fast_mst[k]);
}


void test_loss(const unsigned int n_words, bool random)
{

    dy::ParameterCollection p;

    vector<int> y_true(n_words, 0);
    y_true[0] = -1;

    dy::Parameter x_par;

    if (random)
    {
        std::cout << "Using random weights"
                  << "********************" << std::endl;
        x_par = p.add_parameters({n_words, n_words - 1}, 0, "x");
    }
    else
    {
        std::cout << "Using own weights"
                  << "*****************" << std::endl;

        x_par = p.add_parameters({n_words, n_words - 1}, 0, "x");
        float x = 0.5;
        float y = -0.5;
        auto T = dy::mat(*x_par.values());
        for (int h = 0; h < n_words; ++h)
            for (int m = 1; m < n_words; ++m)
            {
                T(h, m - 1) = (y_true[m] == h ? x : y);
                x += 0.01;
                y += 0.01;
            }
    }

    {
        dy::ComputationGraph cg;
        auto x = dy::parameter(cg, x_par);
        std::cout << x.value() << std::endl << std::endl;
    }
    std::vector<std::string> methods = { "map", "sparsemap", "marginal" };

    for (int method = 0; method < 3; ++method)
    {
        for (int cost_augment = 0; cost_augment < 2; ++cost_augment)
        {
            std::cout << "\n\nTesting " << methods[method]
                      << ", cost_aug=" << cost_augment << std::endl;

            dy::ComputationGraph cg;
            auto x = dy::parameter(cg, x_par);

            Expression y;

            if (cost_augment)
            {
                if (method == 0)
                    y = struct_hinge_loss_tree(x, n_words, y_true);
                else if (method == 1)
                    y = sparsemap_hinge_loss_tree(x, n_words, y_true, 1000);
                else
                {
                    std::cout << "not implemented" << std::endl;
                    return;
                }

            }
            else
            {
                if (method == 0)
                    y = perceptron_loss_tree(x, n_words, y_true);
                else if (method == 1)
                    y = sparsemap_loss_tree(x, n_words, y_true, 1000);
                else if (method == 2)
                    y = marginal_loss_tree(x, n_words, y_true);
            }

            std::cout << dy::as_scalar(cg.forward(y)) << endl;
            cg.backward(y);
            check_grad(p, y, 1);
        }
    }
}


int main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);
    dy::initialize(dyparams);

    cout << "Test fast reparametrization" << endl;
    test_equivalent(41, 20);

    test_loss(5, false);
    test_loss(5, true);
}
