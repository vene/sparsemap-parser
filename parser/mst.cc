#include <vector>
#include <dynet/tensor-eigen.h>
#include <ad3/GenericFactor.h>
#include <ad3/FactorGraph.h>
#include "mst.h"

using std::vector;


void FactorTreeFast::Initialize(int length)
{
    length_ = length;
    index_arcs_.assign(length, vector<int>(length, -1));
    int k = 0;
    for (int m = 1; m < length; m++)
    {
        for (int h = 0; h < length; ++h)
        {
            if (h != m)
            {
                index_arcs_[h][m] = k;
            }
            ++k;
        }
    }
}

vector<int> mst_decode(int length, float* start, float* end)
{
    FactorTreeFast tree_factor;
    tree_factor.Initialize(length); vector<double> unaries_in(start, end);
    double val;
    vector<int> out;
    tree_factor.RunCLE(unaries_in, &out, &val);

    return out;
}


vector<int> mst_decode(dy::Tensor edge_score)
{
    auto flat = dy::vec(edge_score);
    return mst_decode(edge_score.d[0],
                      flat.data(),
                      flat.data() + flat.size());
}


vector<int> mst_cost_aug_decode(dy::Tensor edge_score, vector<int> heads)
{
    // std::cout << edge_score << std::endl << std::endl;
    auto edge_score_t = dy::t<2>(edge_score);
    for(int m = 1; m < heads.size(); ++m)
        for(int h = 0; h < heads.size(); ++h)
            if (heads[m] != h)
                edge_score_t(h, m - 1) += 1;
    // std::cout << edge_score << std::endl << std::endl;

    return mst_decode(edge_score);

}

std::ostream& operator << (std::ostream &o, const SparsemapSolution &sol)
{
    for (auto && val : sol.arc_posteriors)
        o << val << ' ';
    o << '\t';

    for (size_t k = 0; k < sol.active_set.size(); ++k)
    {
        o << sol.distribution[k] << ' ';
        for (auto && head : sol.active_set[k])
            o << head << ' ';
        o << '\t';
    }
    return o;
}

SparsemapSolution sparsemap_decode(dy::Tensor edge_score, size_t max_iter)
{
    auto length = edge_score.d[0];
    auto xvec = vec(edge_score);
    AD3::FactorGraph factor_graph;
    FactorTreeFast tree_factor;
    vector<AD3::BinaryVariable*> vars;

    for (int m = 1; m <= length; ++m)
        for (int h = 0; h <= length; ++h)
            if (h != m)
                vars.push_back(factor_graph.CreateBinaryVariable());

    factor_graph.DeclareFactor(&tree_factor, vars, false);
    tree_factor.Initialize(length);
    tree_factor.SetQPMaxIter(max_iter);
    tree_factor.SetClearCache(false);
    vector<double> unaries_in(xvec.data(), xvec.data() + xvec.size());
    vector<double> adds;
    SparsemapSolution sol;
    tree_factor.SolveQP(unaries_in, adds, &sol.arc_posteriors, &adds);
    sol.distribution = tree_factor.GetQPDistribution();

    vector<AD3::Configuration> active_set = tree_factor.GetQPActiveSet();
    for (auto cfg : active_set)
    {
        auto cfg_int = static_cast<vector<int>*>(cfg);
        sol.active_set.push_back(*cfg_int);
    }
    return sol;
}
