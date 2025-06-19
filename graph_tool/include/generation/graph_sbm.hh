// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2024 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 3 of the License, or (at your option) any
// later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#ifndef GRAPH_SBM_HH
#define GRAPH_SBM_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "sampler.hh"
#include "urn_sampler.hh"

#include "random.hh"

#include "hash_map_wrap.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <bool micro_deg, class Graph, class VProp, class IVec, class FVec,
          class VDProp, class RNG>
void gen_sbm(Graph& g, VProp b, IVec& rs, IVec& ss, FVec probs, VDProp in_deg,
             VDProp out_deg, bool micro_ers, RNG& rng)
{
    typedef typename std::conditional_t<micro_deg,size_t,double> dtype;
    vector<vector<size_t>> rvs;
    vector<vector<dtype>> v_in_probs, v_out_probs;
    for (auto v : vertices_range(g))
    {
        size_t r = b[v];
        if (r >= v_out_probs.size())
        {
            if (graph_tool::is_directed(g))
                v_in_probs.resize(r+1);
            v_out_probs.resize(r+1);
            rvs.resize(r+1);
        }
        rvs[r].push_back(v);
        if (graph_tool::is_directed(g))
            v_in_probs[r].push_back(in_deg[v]);
        v_out_probs[r].push_back(out_deg[v]);
    }

    typedef std::conditional_t<micro_deg,
                               UrnSampler<size_t, false>,
                               Sampler<size_t>> vsampler_t;
    vector<vsampler_t> v_in_sampler_, v_out_sampler;
    for (size_t r = 0; r < rvs.size(); ++r)
    {
        if (graph_tool::is_directed(g))
            v_in_sampler_.emplace_back(rvs[r], v_in_probs[r]);
        v_out_sampler.emplace_back(rvs[r], v_out_probs[r]);
    }

    auto& v_in_sampler = (graph_tool::is_directed(g)) ? v_in_sampler_ : v_out_sampler;

    for (size_t i = 0; i < rs.shape()[0]; ++i)
    {
        size_t r = rs[i];
        size_t s = ss[i];
        auto p = probs[i];

        if (p == 0)
            continue;

        if (!graph_tool::is_directed(g) && r == s)
            p /= 2;

        auto& r_sampler = v_out_sampler[r];
        auto& s_sampler = v_in_sampler[s];

        size_t mrs;
        if (micro_ers)
        {
            mrs = p;
        }
        else
        {
            std::poisson_distribution<> poi(p);
            mrs = poi(rng);
        }

        size_t ers = (&r_sampler != &s_sampler) ? mrs : 2 * mrs;
        if (micro_deg && (!r_sampler.has_n(ers) || !s_sampler.has_n(ers)))
            throw GraphException("Inconsistent SBM parameters: node degrees do not agree with matrix of edge counts between groups");

        for (size_t j = 0; j < mrs; ++j)
        {
            size_t u = r_sampler.sample(rng);
            size_t v = s_sampler.sample(rng);
            add_edge(u, v, g);
        }
    }
}


} // graph_tool namespace

#endif // GRAPH_SBM_HH
