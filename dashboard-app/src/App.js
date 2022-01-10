import "./App.css";

import { io } from "socket.io-client";
import { useState } from "react";
import TrainingPlots from "./components/TrainingPlots";
import NavTabs from "./components/NavTabs";
import { Route, Routes } from "react-router-dom";
import Samples from "./components/Samples";
import { Box } from "@mui/material";
import { MemoryRouter } from "react-router-dom";

function App() {
  const socket = io("/");
  const [data, setData] = useState([]);

  socket.on("data", (new_data) => {
    setData([...data, new_data]);
    console.log(data);
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
            <Route path="/samples" element={<Samples></Samples>}></Route>
          </Routes>
        </Box>
      </MemoryRouter>
    </div>
  );
}

export default App;
