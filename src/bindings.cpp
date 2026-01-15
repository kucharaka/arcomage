#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(arkomag_cpp, m) {
    py::class_<ArcomageEngine>(m, "Engine")
        .def(py::init<>())

        .def("reset", &ArcomageEngine::reset)
        .def("ensure_turn_begun", &ArcomageEngine::ensureTurnBegun)

        .def("current_player", [](ArcomageEngine& e){ return e.st.current; })
        .def("done", [](ArcomageEngine& e){ return e.st.done; })
        .def("winner", [](ArcomageEngine& e){ return e.st.winner; })

        .def("hand_size", &ArcomageEngine::handSize)
        .def("action_size", &ArcomageEngine::actionSize)
        .def("card_count", &ArcomageEngine::cardCount)

        .def("get_observation", &ArcomageEngine::getObservation)
        .def("get_action_mask", &ArcomageEngine::getActionMask)
        .def("step", [](ArcomageEngine& e, int action) {
            e.step(action);
        });

}
