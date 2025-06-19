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

#ifndef CONTINGENCY_HH
#define CONTINGENCY_HH

#include "config.h"

#include <vector>

#include "idx_map.hh"

namespace graph_tool
{

template <bool log_sum, class Graph, class PMap, class VMap, class EMap, class VX, class VY>
void get_contingency_graph(Graph& g, PMap&& partition, VMap&& label, EMap&& mrs,
                           VX& x, VY& y)
{
    typedef typename graph_traits<Graph>::vertex_descriptor vertex_t;
    typedef typename VX::value_type xv_t;
    typedef typename VY::value_type yv_t;

    idx_map<int, vertex_t> x_vertices, y_vertices;

    auto get_v =
        [&](auto& vs, auto val, auto pval)
        {
            auto iter = vs.find(val);
            if (iter == vs.end())
            {
                auto v = add_vertex(g);
                vs[val] = v;
                partition[v] = pval;
                return v;
            }
            return iter->second;
        };

    for (auto& r : x)
    {
        if constexpr (std::is_scalar_v<xv_t>)
        {
            if (r != -1)
                label[get_v(x_vertices, r, 0)] = r;
        }
        else
        {
            for (auto& rn : r)
                label[get_v(x_vertices, rn.first, 0)] = rn.first;
        }
    }

    for (auto& s : y)
    {
        if constexpr (std::is_scalar_v<yv_t>)
        {
            if (s != -1)
                label[get_v(y_vertices, s, 1)] = s;
        }
        else
        {
            for (auto& sn : s)
                label[get_v(y_vertices, sn.first, 1)] = sn.first;
        }
    }

    auto add_mrs
        = [&](auto i, auto u, auto c)
          {
              if constexpr (std::is_scalar_v<yv_t>)
              {
                  auto s = y[i];
                  if (s == -1)
                      return;

                  auto v = get_v(y_vertices, s, 1);
                  auto e = edge(u, v, g);
                  if (!e.second)
                      e = add_edge(u, v, g);
                  mrs[e.first] += c;
              }
              else
              {
                  for (auto& sn : y[i])
                  {
                      auto v = get_v(y_vertices, sn.first, 1);
                      auto e = edge(u, v, g);
                      if (!e.second)
                          e = add_edge(u, v, g);
                      if constexpr (log_sum)
                          mrs[e.first] += (lgamma_fast(sn.second + c + 1) -
                                           lgamma_fast(sn.second + 1) -
                                           lgamma_fast(c + 1));
                      else
                          mrs[e.first] += sn.second + c;
                  }
              }
          };

    for (size_t i = 0; i < x.size(); ++i)
    {
        if constexpr (std::is_scalar_v<xv_t>)
        {
            auto r = x[i];
            if (r == -1)
                continue;
            auto u = get_v(x_vertices, r, 0);
            add_mrs(i, u, 1);
        }
        else
        {
            for (auto& rn : x[i])
            {
                auto u = get_v(x_vertices, rn.first, 0);
                add_mrs(i, u, rn.second);
            }
        }
    }
}
}

#endif // CONTINGENCY_HH
