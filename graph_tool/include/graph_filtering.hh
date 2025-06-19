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

#ifndef FILTERING_HH
#define FILTERING_HH

#include "graph.hh"
#include <boost/version.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/logical.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/assert.hpp>

#include "graph_adaptor.hh"
#include "graph_filtered.hh"
#include "graph_reverse.hh"
#include "graph_util.hh"
#include "mpl_nested_loop.hh"
#include "gil_release.hh"

#include <type_traits>

namespace graph_tool
{

// Graph filtering
// ---------------
//
// We want to generate versions of a template algorithm for every possible type
// of graph views. The types of graph views are the following:
//
//    - The original directed multigraph
//
//    - Filtered graphs, based on MaskFilter below
//
//    - A reversed view of each directed graph (original + filtered)
//
//    - An undirected view of each directed (unreversed) graph (original +
//      filtered)
//
// The total number of graph views is then: 1 + 1 + 2 + 2 = 6
//
// The specific specialization can be called at run time (and generated at
// compile time) with the run_action() function, which takes as arguments the
// GraphInterface worked on, and the template functor to be specialized, which
// must take as first argument a pointer to a graph type, which itself must be a
// template parameter. Additionally, the function can be called optionally with
// up to 4 extra arguments, which must be MPL sequence instances corresponding
// to the type range of the extra arguments which must be passed to the template
// functor. The run_action() will not run the algorithm itself, but will instead
// return a functor (graph_action) which must be called either with no
// arguments, or with parameters of type boost::any which hold internally the
// type and values of the extra parameters to be passed to the action functor,
// and which will define the specialization to be chosen.
//
// Example:
//
// struct my_algorithm
// {
//     template <class Graph, class ValueType>
//     void operator()(Graph& g, ValueType val) const
//     {
//         ... do something ...
//     }
// };
//
// ...
//
// GraphInterface g;
// typedef mpl::vector<int, double, string> value_types;
// double foo = 42.0;
// run_action(g, my_algorithm(), value_types)(boost::any(foo));
//
// The above line will run my_algorithm::operator() with Graph being the
// appropriate graph view type and ValueType being 'double' and val = 42.0.

// Whenever no implementation is called, the following exception is thrown
class ActionNotFound: public GraphException
{
public:
    ActionNotFound(const std::type_info& action,
                   const std::vector<const std::type_info*>& args);
    virtual ~ActionNotFound() throw () {}
private:
    const std::type_info& _action;
    std::vector<const std::type_info*> _args;
};

namespace detail
{

// Implementation
// --------------
//
// The class MaskFilter below is the main filter predicate for the filtered
// graph view, based on descriptor property maps.  It filters out edges or
// vertices which are masked according to a property map with bool (actually
// uint8_t) value type.

template <class DescriptorProperty>
class MaskFilter
{
public:
    typedef typename boost::property_traits<DescriptorProperty>::value_type value_t;
    MaskFilter(){}
    MaskFilter(DescriptorProperty& filtered_property, bool& invert)
        : _filtered_property(&filtered_property), _invert(&invert) {}

    template <class Descriptor>
    inline bool operator() (Descriptor&& d) const
    {
        // ignore if masked

        return get(*_filtered_property, std::forward<Descriptor>(d)) ^ *_invert;

        // This is a critical section. It will be called for every vertex or
        // edge in the graph, every time they're iterated through.
    }

    DescriptorProperty& get_filter() { return *_filtered_property; }
    bool is_inverted() { return *_invert; }

private:
    DescriptorProperty* _filtered_property;
    bool* _invert;
};


// Metaprogramming
// ---------------
//
// We need to generate a type sequence with all the filtered graph views, which
// will be called all_graph_views.


// metafunction class to get the correct filter predicate
template <class Property>
struct get_predicate
{
    typedef MaskFilter<Property> type;
};

template <>
struct get_predicate<boost::keep_all>
{
    typedef boost::keep_all type;
};

// metafunction to get the filtered graph type
struct graph_filter
{
    template <class Graph, class EdgeProperty, class VertexProperty>
    struct apply
    {

        typedef typename get_predicate<EdgeProperty>::type edge_predicate;
        typedef typename get_predicate<VertexProperty>::type vertex_predicate;

        typedef boost::filt_graph<Graph,
                                  edge_predicate,
                                  vertex_predicate> filtered_graph_t;

        // If both predicates are keep_all, then return the original graph
        // type. Otherwise return the filtered_graph type.
        typedef typename boost::mpl::if_<
            typename boost::mpl::and_<
                std::is_same<edge_predicate,
                             boost::keep_all>,
                std::is_same<vertex_predicate,
                             boost::keep_all>
                >::type,
            Graph,
            filtered_graph_t>::type type;
    };
};

// metafunction to get the undirected graph type
struct graph_undirect
{
    template <class Graph>
    struct apply
    {
        typedef boost::undirected_adaptor<Graph> type;
    };
};

// metafunction to get the reversed graph type
struct graph_reverse
{
    template <class Graph>
    struct apply
    {
        typedef boost::reversed_graph<Graph> type;
    };
};


// metafunction class to get the correct property map type
template <class Scalar, class IndexMap>
struct get_property_map_type
{
    typedef typename property_map_type::apply<Scalar, IndexMap>
        ::type::unchecked_t type;
};

template <class IndexMap>
struct get_property_map_type<boost::keep_all, IndexMap>
{
    typedef boost::keep_all type;
};

// this metafunction returns a filtered graph type
struct get_graph_filtered
{
    template <class Graph>
    struct apply
    {
        // if the 'scalar' is the index map itself, use simply that, otherwise
        // get the specific property map
        typedef typename get_property_map_type<
            uint8_t,
            GraphInterface::edge_index_map_t>::type edge_property_map;

        typedef typename get_property_map_type<
            uint8_t,
            GraphInterface::vertex_index_map_t>::type vertex_property_map;

        typedef typename graph_filter::apply<Graph,
                                             edge_property_map,
                                             vertex_property_map>::type type;
    };
};

// this metafunction returns all the possible graph views
struct get_all_graph_views
{
    template <class FiltType,
              class AlwaysDirected = boost::mpl::bool_<false>,
              class NeverDirected = boost::mpl::bool_<false>,
              class AlwaysReversed = boost::mpl::bool_<false>,
              class NeverReversed = boost::mpl::bool_<false>,
              class NeverFiltered = boost::mpl::bool_<false> >
    struct apply
    {

        struct base_graphs:
            boost::mpl::vector1<GraphInterface::multigraph_t> {};

        // reversed graphs
        struct reversed_graphs:
            boost::mpl::if_<AlwaysReversed,
                            typename boost::mpl::transform<base_graphs,
                                                           graph_reverse>::type,
                            typename boost::mpl::if_<
                                NeverReversed,
                                base_graphs,
                                typename boost::mpl::transform<
                                    base_graphs,
                                    graph_reverse,
                                    boost::mpl::back_inserter<base_graphs>
                                    >::type
                                >::type
                            >::type {};

        // undirected graphs
        struct undirected_graphs:
            boost::mpl::if_<AlwaysDirected,
                            reversed_graphs,
                            typename boost::mpl::if_<
                                NeverDirected,
                                typename boost::mpl::transform<
                                    base_graphs,
                                    graph_undirect,
                                    boost::mpl::back_inserter<boost::mpl::vector0<>>
                                    >::type,
                                typename boost::mpl::transform<
                                    base_graphs,
                                    graph_undirect,
                                    boost::mpl::back_inserter<reversed_graphs>
                                    >::type
                                >::type
                            >::type {};

        // filtered graphs
        struct filtered_graphs:
            boost::mpl::if_<NeverFiltered,
                            undirected_graphs,
                            typename boost::mpl::transform<
                                undirected_graphs,
                                get_graph_filtered,
                                boost::mpl::back_inserter<undirected_graphs>>::type
                            >::type {};

        typedef filtered_graphs type;
    };
};

typedef uint8_t filt_scalar_type;

// finally, this type should hold all the possible graph views
struct all_graph_views:
    get_all_graph_views::apply<filt_scalar_type>::type {};

// restricted graph views
struct always_directed:
    get_all_graph_views::apply<filt_scalar_type,boost::mpl::bool_<true> >::type {};

struct never_directed:
    get_all_graph_views::apply<filt_scalar_type,boost::mpl::bool_<false>,
                               boost::mpl::bool_<true> >::type {};

struct always_reversed:
    get_all_graph_views::apply<filt_scalar_type,boost::mpl::bool_<true>,
                               boost::mpl::bool_<false>,boost::mpl::bool_<true> >::type {};

struct never_reversed:
    get_all_graph_views::apply<filt_scalar_type,boost::mpl::bool_<false>,
                               boost::mpl::bool_<false>,boost::mpl::bool_<false>,
                               boost::mpl::bool_<true> >::type {};

struct always_directed_never_reversed:
    get_all_graph_views::apply<filt_scalar_type,boost::mpl::bool_<true>,
                               boost::mpl::bool_<false>,boost::mpl::bool_<false>,
                               boost::mpl::bool_<true> >::type {};

struct never_filtered:
    get_all_graph_views::apply<filt_scalar_type,boost::mpl::bool_<false>,
                               boost::mpl::bool_<false>,boost::mpl::bool_<false>,
                               boost::mpl::bool_<false>,boost::mpl::bool_<true> >::type {};

struct never_filtered_never_reversed:
    get_all_graph_views::apply<filt_scalar_type,boost::mpl::bool_<false>,
                               boost::mpl::bool_<false>,boost::mpl::bool_<false>,
                               boost::mpl::bool_<true>,boost::mpl::bool_<true> >::type {};

struct always_directed_never_filtered_never_reversed:
    get_all_graph_views::apply<filt_scalar_type,boost::mpl::bool_<true>,
                               boost::mpl::bool_<false>,boost::mpl::bool_<false>,
                               boost::mpl::bool_<true>,boost::mpl::bool_<true> >::type {};

struct never_directed_never_filtered_never_reversed:
    get_all_graph_views::apply<filt_scalar_type,boost::mpl::bool_<false>,
                               boost::mpl::bool_<true>,boost::mpl::bool_<false>,
                               boost::mpl::bool_<true>,boost::mpl::bool_<true> >::type {};

// sanity check
typedef boost::mpl::size<all_graph_views>::type n_views;
BOOST_MPL_ASSERT_RELATION(n_views::value, == , boost::mpl::int_<6>::value);

// run_action() and gt_dispatch() implementation
// =============================================

// wrap action to be called, to deal with property maps, i.e., return version
// with no bounds checking.
template <class Action, class Wrap>
struct action_wrap
{
    action_wrap(Action a, bool gil_release) : _a(a), _gil_release(gil_release) {}

    template <class Type, class IndexMap>
    auto& uncheck(boost::checked_vector_property_map<Type,IndexMap>& a,
                  boost::mpl::true_) const
    {
        return a;
    }

    template <class Type, class IndexMap>
    auto uncheck(boost::checked_vector_property_map<Type, IndexMap>& a,
                 boost::mpl::false_) const
    {
        return a.get_unchecked();
    }

    template <class Type>
    auto uncheck(scalarS<Type>& a, boost::mpl::false_) const
    {
        auto pmap = uncheck(a._pmap, boost::mpl::false_());
        return scalarS<decltype(pmap)>(pmap);
    }

    //no op
    template <class Type, class DoWrap>
    Type& uncheck(Type&& a, DoWrap) const { return a; }

    template <class Type>
    auto& deference(Type* a) const
    {
        typedef typename std::remove_const<Type>::type type_t;
        typedef typename boost::mpl::find<detail::all_graph_views, type_t>::type iter_t;
        typedef typename boost::mpl::end<detail::all_graph_views>::type end_t;
        return deference_dispatch(a, typename std::is_same<iter_t, end_t>::type());
    }

    template <class Type>
    auto& deference_dispatch(Type*& a, std::true_type) const
    {
        return a;
    }

    template <class Type>
    Type& deference_dispatch(Type* a, std::false_type) const
    {
        return *a;
    }

    template <class Type>
    Type& deference(Type&& a) const
    {
        return a;
    }

    template <class... Ts>
    void operator()(Ts&&... as) const
    {
        GILRelease gil(_gil_release);
        _a(deference(uncheck(std::forward<Ts>(as), Wrap()))...);
    }

    Action _a;
    bool _gil_release;
};

//
// action_dispatch machinery
//

// a lightweight class for holding a list of types
template <class... T>
struct typelist {};

template <class... T>
constexpr auto to_typelist(std::tuple<T...>) -> typelist<T...>;

template <class Tuple>
using to_typelist_t = decltype(to_typelist(std::declval<Tuple>()));

// handling one typelist/value
// select a binding from the current list
template<class    F,                               // function to bind
         class... Ts,                              // current typelist
         class... TRS,                             // remaining typelists
         class    Arg,
         class... Args>
bool dispatch_loop(F&&                       f,
                   typelist<typelist<Ts...>, TRS...>,
                   Arg&&                     arg,
                   Args&&...                 args) // remaining args
{
    using namespace boost::mpl;

    // determine which one of the Ts we are looking at
    // then recurse with the first argument of F bound accordingly

    void *farg;         // pointer to extracted value from boost::any
                        // we always will know what its type is

    if constexpr (sizeof...(TRS) == 0)
    {
        // just one argument remains to be bound; iterate over types, trying each
        return (((farg = boost::any_cast<Ts>(&arg))
                 ? (f(*static_cast<Ts*>(farg)), true)
                 // try reference_wrapper instead
                 : ((farg = boost::any_cast<std::reference_wrapper<Ts>>(&arg))
                    ? (f(static_cast<std::reference_wrapper<Ts>*>(farg)->get()),
                       true)
                    : false)) || ...);
    }
    else
    {
        // helper function for setting up recursion
        auto dl =
            [&f](auto* a,        // extracted value from boost::any
                 auto&&... args)  // boost::any's yet to be processed
            {
                // create the new F with N-1 arguments
                return dispatch_loop
                    ([&f, a](auto &&... fargs)
                     {
                         f(*a, std::forward<decltype(fargs)>(fargs)...);
                     },
                     typelist<TRS...>{},
                     std::forward<decltype(args)>(args)...);
            };

        return (((farg = boost::any_cast<Ts>(&arg))
                 ? dl(static_cast<Ts*>(farg),
                      std::forward<Args>(args)...)
                 : ((farg = boost::any_cast<std::reference_wrapper<Ts>>(&arg))
                    ? dl(&static_cast<std::reference_wrapper<Ts>*>(farg)->get(),
                         std::forward<Args>(args)...)
                    : false))
                || ...);                           // iterate over Ts...
    }
}

// this takes a functor and type ranges, locates the correct combination
// from the boost::any parameters, and calls the correct function
template <class Action, class Wrap, class... TRS>
struct action_dispatch
{
    action_dispatch(Action a, bool gil_release=true) : _a(a, gil_release) {}

    template <class... Args>
    void operator()(Args&&... args) const
    {
        using namespace boost::mpl;

        bool found = dispatch_loop(_a, typelist<to_typelist_t<to_tuple_t<TRS>>...>{},
                                   std::forward<Args>(args)...);

        if (!found)
        {
            std::vector<const std::type_info*> args_t = {(&(args).type())...};
            throw ActionNotFound(typeid(Action), args_t);
        }
    }

    action_wrap<Action, Wrap> _a;
};

} // details namespace

// dispatch "Action" across all type combinations
template <class GraphViews = detail::all_graph_views,
          class Wrap = boost::mpl::false_>
struct run_action
{
    run_action(bool gil_release=true)
        : _gil_release(gil_release) {};

    template <class Action, class... TRS>
    auto operator()(GraphInterface& gi, Action a, TRS...)
    {
        auto dispatch = detail::action_dispatch<Action,Wrap,GraphViews,TRS...>(a, _gil_release);
        auto wrap = [dispatch, &gi](auto&&... args) { dispatch(gi.get_graph_view(), args...); };
        return wrap;
    }

    bool _gil_release;
};

template <class Wrap = boost::mpl::false_>
struct gt_dispatch
{
    gt_dispatch(bool gil_release=true)
        : _gil_release(gil_release) {};

    template <class Action, class... TRS>
    auto operator()(Action a, TRS...)
    {
        return detail::action_dispatch<Action,Wrap,TRS...>(a, _gil_release);
    }

    bool _gil_release;
};

typedef detail::all_graph_views all_graph_views;
typedef detail::always_directed always_directed;
typedef detail::never_directed never_directed;
typedef detail::always_reversed always_reversed;
typedef detail::never_reversed never_reversed;
typedef detail::always_directed_never_reversed always_directed_never_reversed;
typedef detail::never_filtered never_filtered;
typedef detail::never_filtered_never_reversed never_filtered_never_reversed;
typedef detail::always_directed_never_filtered_never_reversed always_directed_never_filtered_never_reversed;
typedef detail::never_directed_never_filtered_never_reversed never_directed_never_filtered_never_reversed;


// returns true if graph filtering was enabled at compile time
bool graph_filtering_enabled();


template <class Graph, class GraphInit>
std::shared_ptr<Graph> get_graph_ptr(GraphInterface& gi, GraphInit&, std::true_type)
{
    return gi.get_graph_ptr();
}

template <class Graph, class GraphInit>
std::shared_ptr<Graph> get_graph_ptr(GraphInterface&, GraphInit& g, std::false_type)
{
    return std::make_shared<Graph>(g);
}

// this function retrieves a graph view stored in graph_views, or stores one if
// non-existent
template <class Graph>
typename std::shared_ptr<Graph>
retrieve_graph_view(GraphInterface& gi, Graph& init)
{
    typedef typename std::remove_const<Graph>::type g_t;
    size_t index = boost::mpl::find<detail::all_graph_views,g_t>::type::pos::value;
    auto& graph_views = gi.get_graph_views();
    if (index >= graph_views.size())
        graph_views.resize(index + 1);
    std::shared_ptr<void>& gview = graph_views[index];
    if (!gview)
    {
        std::shared_ptr<g_t> new_g =
            get_graph_ptr<g_t>(gi, init,
                               std::is_same<g_t, GraphInterface::multigraph_t>());
        gview = new_g;
    }
    return std::static_pointer_cast<g_t>(gview);
}

} //graph_tool namespace

// Overload add_vertex() and add_edge() to filtered graphs, so that the new
// descriptors are always valid

namespace boost
{
template <class Graph, class EdgeProperty, class VertexProperty>
auto
add_vertex(boost::filt_graph<Graph,
                             graph_tool::detail::MaskFilter<EdgeProperty>,
                             graph_tool::detail::MaskFilter<VertexProperty>>& g)
{
    auto v = add_vertex(const_cast<Graph&>(g._g));
    auto& filt = g._vertex_pred.get_filter();
    auto cfilt = filt.get_checked();
    cfilt[v] = !g._vertex_pred.is_inverted();
    return v;
}

template <class Graph, class EdgeProperty, class VertexProperty, class Vertex>
auto
add_edge(Vertex s, Vertex t, filt_graph<Graph,
                                        graph_tool::detail::MaskFilter<EdgeProperty>,
                                        graph_tool::detail::MaskFilter<VertexProperty>>& g)
{
    auto e = add_edge(s, t, const_cast<Graph&>(g._g));
    auto& filt = g._edge_pred.get_filter();
    auto cfilt = filt.get_checked();
    cfilt[e.first] = !g._edge_pred.is_inverted();
    return e;
}

// Used to skip filtered vertices

template <class Graph, class EP, class VP, class Vertex>
bool is_valid_vertex(Vertex v, const boost::filt_graph<Graph,EP,VP>& g)
{
    return (v < num_vertices(g) &&
            vertex(v, g) != graph_traits<boost::filt_graph<Graph,EP,VP>>::null_vertex());
}

template <class Graph, class Vertex>
bool is_valid_vertex(Vertex v, const boost::reversed_graph<Graph>& g)
{
    return is_valid_vertex(v, g._g);
}

template <class Graph, class Vertex>
bool is_valid_vertex(Vertex v, const boost::undirected_adaptor<Graph>& g)
{
    return is_valid_vertex(v, g.original_graph());
}

template <class Graph, class Vertex>
bool is_valid_vertex(Vertex v, const Graph& g)
{
    return v < num_vertices(g);
}


} // namespace boost

#endif // FILTERING_HH
