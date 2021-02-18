#pragma once

#include <vector>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/nodes-def-macros.h>
#include <dynet/tensor-eigen.h>

#include "mst.h"


namespace dy = dynet;
using namespace dynet;


struct MAPLossTree : public dy::Node
{
    size_t length_;
    bool cost_augment_;
    std::vector<int> y_true_;
    explicit MAPLossTree(const std::initializer_list<dy::VariableIndex>& a,
                         size_t length,
                         std::vector<int> y_true,
                         bool cost_augment)
        : dy::Node(a), length_(length), cost_augment_(cost_augment), y_true_(y_true)
    {
        this->has_cuda_implemented = false;
    }
    DYNET_NODE_DEFINE_DEV_IMPL()
    size_t aux_storage_size() const override;
};


struct SparseMAPLossTree : public MAPLossTree
{
    size_t max_iter_;
    int print_;
    explicit SparseMAPLossTree(const std::initializer_list<dy::VariableIndex>& a,
                               size_t length,
                               std::vector<int> y_true,
                               bool cost_augment,
                               size_t max_iter,
                               int print=0)
        : MAPLossTree(a, length, y_true, cost_augment), max_iter_(max_iter)
        , print_(print)
    {
        this->has_cuda_implemented = false;
    }
    DYNET_NODE_DEFINE_DEV_IMPL()
    size_t aux_storage_size() const override;
};


struct MarginalLossTree : public MAPLossTree
{
    explicit MarginalLossTree(const std::initializer_list<dy::VariableIndex>& a,
                              size_t length,
                              std::vector<int> y_true,
                              bool cost_augment)
        : MAPLossTree(a, length, y_true, cost_augment)
    {
        this->has_cuda_implemented = false;
    }
    DYNET_NODE_DEFINE_DEV_IMPL()
    size_t aux_storage_size() const override;
};


dy::Expression perceptron_loss_tree(
    const dy::Expression& x,
    size_t length,
    std::vector<int> y_true);


dy::Expression struct_hinge_loss_tree(
    const dy::Expression& x,
    size_t length,
    std::vector<int> y_true);


dy::Expression sparsemap_loss_tree(
    const dy::Expression& x,
    size_t length,
    std::vector<int> y_true,
    size_t max_iter=20,
    int print=0);


dy::Expression sparsemap_hinge_loss_tree(
    const dy::Expression& x,
    size_t length,
    std::vector<int> y_true,
    size_t max_iter=20,
    int print=0);


dy::Expression marginal_loss_tree(
    const dy::Expression& x,
    size_t length,
    std::vector<int> y_true);

