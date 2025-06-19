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

#ifndef GRAPH_PRICE_HH
#define GRAPH_PRICE_HH

#include <boost/functional/hash.hpp>
#include "graph_util.hh"
#include "random.hh"

#include "idx_map.hh"
#include "dynamic_sampler.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

struct get_price
{
    template <class Graph>
    void operator()(Graph& g, size_t N, double gamma, double c, size_t m,
                    rng_t& rng) const
    {
        typedef typename mpl::if_<typename is_directed_::apply<Graph>::type,
                                  in_degreeS, out_degreeS>::type Deg;

        DynamicSampler<typename graph_traits<Graph>::vertex_descriptor> sampler;

        double p;
        for (auto v : vertices_range(g))
        {
            p = pow(Deg()(v, g), gamma) + c;
            if (p < 0)
                throw GraphException("Cannot connect edges: probabilities are negative");
            if (p > 0)
                sampler.insert(v, p);
        }

        if (sampler.empty())
            throw GraphException("Cannot connect edges: seed graph is empty, or has zero probability");

        idx_set<typename graph_traits<Graph>::vertex_descriptor, false, false>
            visited(num_vertices(g));
        for (size_t i = 0; i < N; ++i)
        {
            visited.clear();
            auto v = add_vertex(g);
            for (size_t j = 0; j < std::min(m, sampler.size()); ++j)
            {
                auto w = sampler.sample(rng);

                if (visited.find(w) != visited.end())
                {
                    --j;
                    continue;
                }

                visited.insert(w);
                add_edge(v, w, g);

                p = pow(Deg()(w, g), gamma) + c;

                sampler.remove(w);
                sampler.insert(w, p);
            }
            p = pow(Deg()(v, g), gamma) + c;
            if (p > 0)
                sampler.insert(v, p);
        }
    }
};

} // namespace graph_tool

#endif // GRAPH_PRICE_HH
