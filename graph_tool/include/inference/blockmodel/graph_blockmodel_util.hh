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

#ifndef GRAPH_BLOCKMODEL_UTIL_HH
#define GRAPH_BLOCKMODEL_UTIL_HH

#include "config.h"

#include <any>

#include "graph_blockmodel_entropy.hh"
#include "graph_blockmodel_partition.hh"
#include "graph_blockmodel_entries.hh"
#include "graph_blockmodel_emat.hh"
#include "graph_blockmodel_elist.hh"
#include "graph_blockmodel_weights.hh"

namespace graph_tool
{

class BlockStateVirtualBase {
public:
    virtual ~BlockStateVirtualBase() {};
    virtual double entropy(const entropy_args_t& eargs, bool propagate) = 0;
    virtual void add_partition_node(size_t v, size_t r) = 0;
    virtual void remove_partition_node(size_t v, size_t r) = 0;
    virtual void set_vertex_weight(size_t v, int w) = 0;
    virtual void coupled_resize_vertex(size_t v) = 0;
    virtual BlockStateVirtualBase* get_coupled_state() = 0;
    virtual double virtual_move(size_t v, size_t r, size_t nr,
                                const entropy_args_t& eargs) = 0;
    virtual void sample_branch(size_t v, size_t u, rng_t& rng) = 0;
    virtual void copy_branch(size_t v, BlockStateVirtualBase&) = 0;
    virtual size_t sample_block(size_t v, double c, double d, rng_t& rng) = 0;
    virtual double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                                 bool reverse) = 0;
    virtual size_t add_block(size_t n = 1) = 0;
    virtual void add_edge(const GraphInterface::edge_t& e) = 0;
    virtual void remove_edge(const GraphInterface::edge_t& e) = 0;
    virtual void add_edge_rec(const GraphInterface::edge_t& e) = 0;
    virtual void remove_edge_rec(const GraphInterface::edge_t& e) = 0;
    virtual void update_edge_rec(const GraphInterface::edge_t& e,
                                 const std::vector<double>& delta) = 0;
    virtual void add_edge(size_t u, size_t v, GraphInterface::edge_t& e, int dm) = 0;
    virtual void remove_edge(size_t u, size_t v, GraphInterface::edge_t& e, int dm) = 0;
    virtual double modify_edge_dS(size_t u, size_t v, const GraphInterface::edge_t& e,
                                  int dm, const entropy_args_t& ea) = 0;
    virtual void propagate_delta(size_t u, size_t v,
                                 std::vector<std::tuple<size_t, size_t,
                                             GraphInterface::edge_t, int,
                                             std::vector<double>>>& entries) = 0;
    virtual double propagate_entries_dS(size_t u, size_t v, int du, int dv,
                                        std::vector<std::tuple<size_t, size_t,
                                                               GraphInterface::edge_t, int,
                                                               std::vector<double>>>& entries,
                                        const entropy_args_t& ea,
                                        std::vector<double>& dBdx, int dL) = 0;
    virtual double get_delta_partition_dl(size_t v, size_t r, size_t nr,
                                          const entropy_args_t& ea) = 0;
    virtual vprop_map_t<int32_t>::type::unchecked_t& get_b() = 0;
    virtual vprop_map_t<int32_t>::type::unchecked_t& get_pclabel() = 0;
    virtual vprop_map_t<int32_t>::type::unchecked_t& get_bclabel() = 0;
    virtual bool check_edge_counts(bool emat=true) = 0;
    virtual void check_node_counts() = 0;
    virtual bool allow_move(size_t r, size_t nr) = 0;
    virtual void relax_update(bool relax) = 0;
    virtual BlockStateVirtualBase* deep_copy() = 0;
    virtual BlockStateVirtualBase* deep_copy(boost::any) { return deep_copy(); } ;
    virtual void deep_assign(const BlockStateVirtualBase&) = 0;
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_UTIL_HH
