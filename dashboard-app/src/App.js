import "./App.css";

import { io } from "socket.io-client";
import { useState } from "react";
import TrainingPlots from "./components/TrainingPlots";
import NavTabs from "./components/NavTabs";
import { Route, Routes } from "react-router-dom";
import Samples from "./components/Samples";
import { Box } from "@mui/material";
import { MemoryRouter } from "react-router-dom";
import Landscape from "./components/Landscape";

function App() {
  const socket = io("/");
  const [data, setData] = useState([]);
  const [samples, setSamples] = useState({ samples: [] });
  const [landscape, setLandscape] = useState({ img: null });

  socket.on("data", (new_data) => {
    setData([...data, new_data]);
  });

  socket.on("samples", (new_data) => {
    setSamples(new_data);
  });

  socket.on("landscape", (new_data) => {
    setLandscape(new_data);
  });

  return (
    <div className="App">
      <MemoryRouter initialEntries={["/"]} initialIndex={0}>
        <Box sx={{ width: "100%" }}>
          <NavTabs></NavTabs>
          <Routes>
            <Route
              path="/"
              element={<TrainingPlots data={data}></TrainingPlots>}
            ></Route>
            <Route
              path="/samples"
              element={<Samples data={samples.samples}></Samples>}
            ></Route>
            <Route
              path="/landscape"
              element={
                <Landscape
                  surf={landscape.surf}
                  contour={landscape.contour}
                ></Landscape>
              }
            ></Route>
          </Routes>
        </Box>
      </MemoryRouter>
    </div>
  );
}

export default App;
