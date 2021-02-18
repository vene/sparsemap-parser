#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>
#include <cassert>

using std::vector;

const auto maxln = std::numeric_limits<std::streamsize>::max();


struct Sentence
{
    std::vector<unsigned> tok_ixs;
    std::vector<unsigned> pos_ixs;
    std::vector<int> heads;

    friend std::istream& operator>>(std::istream& in,
                                    Sentence& data);
    friend std::ostream& operator<<(std::ostream& out,
                                    const Sentence& data);

};


std::istream& operator>>(std::istream& in, Sentence& data)
{
    std::string ixs_buf, pos_buf, heads_buf;
    in.ignore(maxln, '\t');  // ignore up to first tab: actual text
    std::getline(in, ixs_buf, '\t');
    std::getline(in, pos_buf, '\t');
    std::getline(in, heads_buf);
    if (!in)  // failed
        return in;

    {
        std::stringstream ixs(ixs_buf);
        unsigned tmp;
        while(ixs >> tmp)
            data.tok_ixs.push_back(tmp);
    }
    {
        std::stringstream ixs(pos_buf);
        unsigned tmp;
        while(ixs >> tmp)
            data.pos_ixs.push_back(tmp);
    }
    {
        std::stringstream heads(heads_buf);
        int tmp;
        while(heads >> tmp)
            data.heads.push_back(tmp);
    }

    if (data.tok_ixs.size() != data.pos_ixs.size())
        throw std::invalid_argument("pos and tokens have different number");

    if (1 + data.tok_ixs.size() != data.heads.size())
        throw std::invalid_argument("heads and tokens have different number");

    return in;

}


std::ostream& operator<<(std::ostream& out,
                         const Sentence& data)
{
    out << "tokens: ";
    for(auto&& i: data.tok_ixs) out << i << " ";
    out << "\n POS tags: ";
    for(auto&& i: data.pos_ixs) out << i << " ";
    out << '\n' << "parse tree: ";
    for(auto&& i: data.heads) out << i << " ";
    out << std::endl;

    return out;
}


vector<vector<Sentence> > read_batches(const std::string& filename,
                                       size_t batch_size)
{

    vector<vector<Sentence> > batches;

    std::ifstream in(filename);
    assert(in);

    vector<Sentence> curr_batch;

    while(in)
    {
        Sentence s;
        in >> s;
        if (!in) break;

        if (curr_batch.size() == batch_size)
        {
            batches.push_back(curr_batch);
            curr_batch.clear();
        }
        curr_batch.push_back(s);
    }

    // leftover batch
    if (curr_batch.size() > 0)
        batches.push_back(curr_batch);

    return batches;
}


// returns counts for each word
std::vector<float> read_vocab(const std::string& filename)
{
    std::ifstream in(filename);
    assert(in);

    std::vector<float> vocab;
    std::string tmp;
    while(in)
    {
        float count;
        in.ignore(maxln, '\t');  // ignore up to first tab: actual word
        in >> count;
        if (!in) break;
        vocab.push_back(count);
    }
    return vocab;
}
