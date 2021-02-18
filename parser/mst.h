#pragma once

#include <vector>
#include <dynet/tensor.h>
#include <parsing/FactorTree.h>

namespace dy = dynet;

struct SparsemapSolution
{
    vector<vector<int> > active_set;
    vector<double> distribution;
    vector<double> arc_posteriors;
};

class FactorTreeFast: public AD3::FactorTree {
 public:
  void Initialize(int length);
};

std::ostream& operator << (std::ostream &o, const SparsemapSolution &args);

std::vector<int> mst_decode(int length, float* start, float* end);
std::vector<int> mst_decode(dy::Tensor edge_score);
std::vector<int> mst_cost_aug_decode(dy::Tensor edge_score, std::vector<int> heads);
SparsemapSolution sparsemap_decode(dy::Tensor edge_score, size_t max_iter);
