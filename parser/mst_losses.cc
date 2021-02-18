#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <ad3/FactorGraph.h>
#include <dynet/nodes-impl-macros.h>

#include "logval.h"
#include "mst_losses.h"

// typedef LogVal<float> LogValf;
typedef LogVal<double> LogVald;

namespace Eigen {
    typedef Eigen::Matrix<LogVald, Dynamic, 1> VectorXlogd;
    typedef Eigen::Matrix<LogVald, Dynamic, Dynamic> MatrixXlogd;
}


using namespace dynet;
namespace ei = Eigen;
using std::string;
using std::vector;


size_t MAPLossTree::aux_storage_size() const
{
    // we store the returned configuration
    return sizeof(int) * length_;
}


size_t SparseMAPLossTree::aux_storage_size() const
{
    // it's easier to store the actual unary posteriors
    return sizeof(double) * length_ * (length_ - 1);
}


size_t MarginalLossTree::aux_storage_size() const
{
    return sizeof(double) * length_ * (length_ - 1);
}


string MAPLossTree::as_string(const vector<string>& arg_names) const
{
    std::ostringstream s;
    s << "map_loss_tree(" << arg_names[0] << ", "
      << "length=" << length_ << ", "
      << "cost_augment=" << cost_augment_ << ")";
    return s.str();
}


string SparseMAPLossTree::as_string(const vector<string>& arg_names) const
{
    std::ostringstream s;
    s << "sparse_map_loss_tree(" << arg_names[0] << ", "
      << "length=" << length_ << ", "
      << "max_iter=" << max_iter_ << ", "
      << "cost_augment=" << cost_augment_ << ")";
    return s.str();
}


string MarginalLossTree::as_string(const vector<string>& arg_names) const
{
    std::ostringstream s;
    s << "marginal_loss_tree(" << arg_names[0] << ", "
      << "length=" << length_ << ", "
      << "cost_augment=" << cost_augment_ << ")";
    return s.str();
}


Dim MAPLossTree::dim_forward(const vector<Dim> &xs) const
{
    DYNET_ARG_CHECK(xs.size() == 1, "tree structured losses take a single input");
    DYNET_ARG_CHECK(xs[0][0] == length_, "input has wrong first dim");
    DYNET_ARG_CHECK(xs[0][1] == length_ - 1, "input has wrong second dim");
    return Dim({1});
}


Dim SparseMAPLossTree::dim_forward(const vector<Dim> &xs) const
{
    return MAPLossTree::dim_forward(xs);
}


Dim MarginalLossTree::dim_forward(const vector<Dim> &xs) const
{
    return MAPLossTree::dim_forward(xs);
}


template<class MyDevice>
void MAPLossTree::forward_dev_impl(const MyDevice& dev,
                                   const vector<const Tensor*>& xs,
                                   Tensor& fx) const
{
    const Tensor* x = xs[0];
    auto xvec = vec(*x);
    auto out = vec(fx);

    FactorTreeFast tree_factor;
    tree_factor.Initialize(length_);
    vector<double> unaries_in(xvec.data(), xvec.data() + xvec.size());

    // evaluate y_true
    vector<double> additionals;
    double true_val;
    std::vector<int>* ptr = const_cast<std::vector<int>*>(&y_true_);
    AD3::Configuration true_cfg = static_cast<AD3::Configuration>(ptr);
    tree_factor.Evaluate(unaries_in, additionals, true_cfg, &true_val);

    // cost-augment
    if (cost_augment_)
    {
        for (auto&& val : unaries_in) val += 1;
        tree_factor.UpdateMarginalsFromConfiguration(true_cfg,
                                                     -1,
                                                     &unaries_in,
                                                     &additionals);

    }

    // predict
    double pred_val;
    vector<int> config;
    tree_factor.RunCLE(unaries_in, &config, &pred_val);

    // std::cout << *x << std::endl;
    // std::cout << "Pred cfg: ";
    // for(auto && yy : config) std::cout << yy << " ";
    // std::cout << std::endl;
    // std::cout << "True cfg: ";
    // for(auto && yy : y_true_) std::cout << yy << " ";
    // std::cout << std::endl;
    // std::cout << "Loss val: " << pred_val - true_val << std::endl << std::endl;

    int* aux = static_cast<int*>(aux_mem);
    std::copy(config.begin(), config.end(), aux);

    out(0) = pred_val - true_val;
}


template <class MyDevice>
void MAPLossTree::backward_dev_impl(const MyDevice& dev,
                                    const vector<const Tensor*>& xs,
                                    const Tensor& fx,
                                    const Tensor& dEdf,
                                    unsigned i,
                                    Tensor& dEdxi) const
{
    int* y_pred = static_cast<int*>(aux_mem);

    auto out = mat(dEdxi);

    for(int m = 1; m < length_; ++m)
    {
        // std::cout << y_pred[m] << " " << y_true_[m] << std::endl;
        out(y_pred[m], m - 1) += 1;
        out(y_true_[m], m - 1) -= 1;
    }
}


DYNET_NODE_INST_DEV_IMPL(MAPLossTree);


template<class MyDevice>
void SparseMAPLossTree::forward_dev_impl(const MyDevice& dev,
                                         const vector<const Tensor*>& xs,
                                         Tensor& fx) const
{
    const Tensor* x = xs[0];
    auto xvec = vec(*x);
    auto out = vec(fx);

    AD3::FactorGraph factor_graph;
    // factor_graph.SetVerbosity(3);
    FactorTreeFast tree_factor;
    vector<AD3::BinaryVariable*> vars;

    for (int m = 1; m <= length_; ++m)
        for (int h = 0; h <= length_; ++h)
            if (h != m)
                vars.push_back(factor_graph.CreateBinaryVariable());

    factor_graph.DeclareFactor(&tree_factor, vars, false);
    tree_factor.Initialize(length_);
    tree_factor.SetQPMaxIter(max_iter_);
    tree_factor.SetClearCache(false);

    vector<double> unaries_in(xvec.data(), xvec.data() + xvec.size());
    vector<double> grad(xvec.size(), 0);
    vector<double> additionals;

    std::vector<int>* ptr = const_cast<std::vector<int>*>(&y_true_);
    AD3::Configuration true_cfg = static_cast<AD3::Configuration>(ptr);

    // cost-augment
    if (cost_augment_)
    {
        for (auto&& val : unaries_in) val += 1;
        tree_factor.UpdateMarginalsFromConfiguration(true_cfg,
                                                     -1,
                                                     &unaries_in,
                                                     &additionals);
    }

    // predict

    tree_factor.SolveQP(unaries_in, additionals, &grad, &additionals);

    if (print_)
    {
        unsigned int nnz_arcs = 0;
        for (auto && val : grad)
            if (val > 0)
                nnz_arcs += 1;

        auto active_set = tree_factor.GetQPActiveSet();
        int n_active = active_set.size();
        std::cerr << n_active << " " << nnz_arcs << " " << vars.size() << " ";
    }
    // for (int k = 0; k < n_active; ++k)
    // {
        // auto cfg = active_set[k];
        // auto cfg_ = static_cast<vector<int>* >(cfg);
        // std::cout << "Predicted cfg #"<< k << ": ";
        // for(auto && yy : *cfg_) std::cout << yy << " ";
        // std::cout << std::endl;
    // }

    vector<double> qp_bak(grad.begin(), grad.end());

    // compute 0.5 ( ||u_true||^2 - ||u*||^2)
    double val = y_true_.size() - 1;  // u_true is a vertex
    for (int k = 0; k < grad.size(); ++k)
        val -= grad[k] * grad[k];

    val *= 0.5;

    // double normdiff = val;

    tree_factor.UpdateMarginalsFromConfiguration(true_cfg,
                                                 -1,
                                                 &grad,
                                                 &additionals);
    // vector<double> truey(xvec.size(), 0);
    // tree_factor.UpdateMarginalsFromConfiguration(true_cfg,
                                                 // 1,
                                                 // &truey,
                                                 // &additionals);

    // compute <v* - v_true, theta>
    // (on the feasible set = <u* - u, theta_u>)
    for (int k = 0; k < xvec.size(); ++k)
    {
        val += grad[k] * unaries_in[k];
    }

    auto grad_mem = static_cast<double*>(aux_mem);
    std::copy(grad.begin(), grad.end(), grad_mem);
    out(0) = val;
    // assert(val >= 0);
    if (val < 0)
    {
        std::cout << "negative value " << val << std::endl;
        //for (auto && yy : grad) std::cout << yy << " ";
        // std::cout << "normdiff " << normdiff << std::endl;
        // std::cout << "unaries " << std::endl;
        // for (auto && yy : unaries_in) std::cout << yy << " ";
        //std::cout << std::endl;
        // std::cout << "begin trueval" << std::endl;
        // double xx = 0;
        // for (int k = 0; k < xvec.size(); ++k)
        // {
            // xx += truey[k] * unaries_in[k];
        // }
        // std::cout << xx << std::endl;
        // std::cout << "end trueval" << std::endl;
        // std::cout << "begin predval" << std::endl;
        // xx = 0;
        // for (int k = 0; k < xvec.size(); ++k)
        // {
            // xx += qp_bak[k] * unaries_in[k];
        // }
        // std::cout << xx << std::endl;
        // std::cout << "end predval" << std::endl;
        // throw 1;
    }

    /*
    // TESTING: explicitly compute value over all configurations
    for (auto && yy : y_true_) std::cout << yy << " ";
    std::cout << "\t";
    auto active_set = tree_factor.GetQPActiveSet();
    int n_active = active_set.size();
    auto distribution = tree_factor.GetQPDistribution();

    // double val2 = 0;
    for (int k = 0; k < n_active; ++k)
    {
        auto cfg = active_set[k];
        auto cfg_ = static_cast<vector<int>* >(cfg);
        std::cout << distribution[k] << ": ";
        for(auto && yy : *cfg_) std::cout << yy << " ";
        std::cout << "\t";
        //double innerval;
        //tree_factor.Evaluate(unaries_in, additionals, cfg, &innerval);
        // val2 += distribution[k] * innerval;
    }
    // double true_val;
    // tree_factor.Evaluate(unaries_in, additionals, true_cfg, &true_val);
    // std::cout << val2 - true_val << " should be the same" << std::endl;
    */
}


template <class MyDevice>
void SparseMAPLossTree::backward_dev_impl(const MyDevice& dev,
                                          const vector<const Tensor*>& xs,
                                          const Tensor& fx,
                                          const Tensor& dEdf,
                                          unsigned i,
                                          Tensor& dEdxi) const
{
    double* grad = static_cast<double*>(aux_mem);

    auto out = vec(dEdxi);
    for(int k = 0; k < out.size(); ++k)
    {
        out(k) += grad[k];
    }
}


DYNET_NODE_INST_DEV_IMPL(SparseMAPLossTree);


template<class MyDevice>
void MarginalLossTree::forward_dev_impl(const MyDevice& dev,
                                        const vector<const Tensor*>& xs,
                                        Tensor& fx) const
{
    const Tensor* x = xs[0];
    auto xmat = mat(*x);
    auto out = vec(fx);

    // if we have a single word, there's no way to get it wrong
    // so we can save some time. We should do this for the other losses.
    if (length_ == 2)
    {
        out(0) = 0;
        float* aux = static_cast<float*>(aux_mem);

        // length * (length_ - 1) = 2 * 1 = 2
        for (int h = 0; h < 2; ++h)
            *(aux++) = 0;
        return;
    }

    ei::Map<ei::MatrixXf> logp(xmat.data(), length_, length_ - 1);

    // subtract a constant and exponentiate
    float C = logp.mean();
    // float C = logp.maxCoeff();

    // root[m] = exp theta_(0, m)
    ei::VectorXlogd root(length_ - 1);
    for (int m = 1; m < length_; ++m)
        root(m - 1) = LogVald::exp(logp(0, m - 1) - C);

    // A[h-1, m-1] = exp theta_(h, m-1), 0 on diag
    ei::MatrixXlogd A(length_ - 1, length_ - 1);
    for (int h = 1; h < length_; ++h)
        for (int m = 1; m < length_; ++m)
            A(h - 1, m - 1) = (h == m) ? LogVald::Zero()  // on diagonal
                                       : LogVald::exp(logp(h, m - 1) - C);

    // laplacian (aka Kirkhoff matrix)
    auto L = A;
    L *= -1;
    L.diagonal() = A.colwise().sum();
    L.diagonal() += root;

    ei::FullPivLU<ei::MatrixXlogd> lu(L);
    auto logdet = lu.determinant().logabs();
    auto Z = logdet + C * (length_ - 1);

    if (Z < -99999)
    {
        // numerical issues have lead to det(L) = 0.
        // We return loss = 0, grad = 0, and warn the user.
        std::cout << "Warning: highly negative Z encountered: "
                  << Z
                  << " for sentence of size "
                  << length_
                  << " with correction constant C= "
                  << C
                  << ". Returning loss=grad=0."
                  << std::endl;

        // set loss = 0
        out(0) = 0;

        // set grad = 0
        float* aux = static_cast<float*>(aux_mem);
        for (int m = 1; m < length_; ++m)
            for (int h = 0; h < length_; ++h)
                *(aux++) = 0;

        #ifdef DEBUG_NEGINF
            std::cout << "Negative value\n"
                      << logdet << std::endl
                      << "C=" << C << std::endl
                      << lu.determinant().as_float() << std::endl
                      << std::endl
                      << logp
                      << std::endl
                      << std::endl
                      << xmat - C
                      << std::endl
                      << std::endl;

            for (int h = 1; h < length_; ++h)
            {
                for (int m = 1; m < length_; ++m)
                    std::cout << L(h - 1, m - 1).as_float() << " ";
                std::cout << std::endl;
            }
            std::cout <<std::endl;
            throw 1;
        #endif


        return;
    }

    // auto Linvt = lu.inverse().transpose();  // DO NOT DO THIS!!!
    ei::MatrixXlogd Linvt = lu.inverse().transpose();

    // marginal posteriors via matrix-tree theorem
    //
    // dlogdet / dlogp = dlogdet/dL * dL / dlogp
    //                 = Linvt * dL / dlogp
    ei::MatrixXlogd mu(length_, length_ - 1);
    mu.row(0) = root.cwiseProduct(Linvt.diagonal());

    // todo vectorize this using eigen
    for (int m = 1; m < length_; ++m)
    {
        for (int h = 1; h < length_; ++h)
        {
            mu(h, m - 1) =
                A(h - 1, m - 1) * (Linvt(m - 1, m - 1) -
                                   Linvt(h - 1, m - 1));
        }
    }

    // compute score of y_true and make mu the gradient
    float true_val = 0;

    for (int m = 1; m < length_; ++m)
    {
        true_val += xmat(y_true_[m], m - 1);
        mu(y_true_[m], m - 1) -= 1;
    }

    // store gradient for backward pass
    // (cannot use std::copy because of log-space class)
    float* aux = static_cast<float*>(aux_mem);
    for (int m = 1; m < length_; ++m)
        for (int h = 0; h < length_; ++h)
            *(aux++) = mu(h, m - 1).as_float();

    out(0) = Z - true_val;
}


template <class MyDevice>
void MarginalLossTree::backward_dev_impl(const MyDevice& dev,
                                         const vector<const Tensor*>& xs,
                                         const Tensor& fx,
                                         const Tensor& dEdf,
                                         unsigned i,
                                         Tensor& dEdxi) const
{
    float* grad = static_cast<float*>(aux_mem);

    auto out = vec(dEdxi);
    for(int k = 0; k < out.size(); ++k)
    {
        out(k) += grad[k];
    }
    // std::cout << "Gradient norm " << out.norm() << std::endl << std::endl;
}


DYNET_NODE_INST_DEV_IMPL(MarginalLossTree);

Expression perceptron_loss_tree(const Expression& x, size_t length, vector<int> y_true)
{
    return Expression(
        x.pg,
        x.pg->add_function<MAPLossTree>({x.i}, length, y_true, false));
}


Expression struct_hinge_loss_tree(const Expression& x, size_t length, vector<int> y_true)
{
    return Expression(
        x.pg,
        x.pg->add_function<MAPLossTree>({x.i}, length, y_true, true));
}

dy::Expression sparsemap_loss_tree(
    const dy::Expression& x,
    size_t length,
    std::vector<int> y_true,
    size_t max_iter,
    int print)
{
    return Expression(
        x.pg,
        x.pg->add_function<SparseMAPLossTree>({x.i},
                                              length,
                                              y_true,
                                              false,
                                              max_iter,
                                              print));
}


dy::Expression sparsemap_hinge_loss_tree(
    const dy::Expression& x,
    size_t length,
    std::vector<int> y_true,
    size_t max_iter,
    int print)
{
    return Expression(
        x.pg,
        x.pg->add_function<SparseMAPLossTree>({x.i},
                                              length,
                                              y_true,
                                              true,
                                              max_iter,
                                              print));
}


Expression marginal_loss_tree(const Expression& x, size_t length, vector<int> y_true)
{
    return Expression(
        x.pg,
        x.pg->add_function<MarginalLossTree>({x.i}, length, y_true, false));
}
