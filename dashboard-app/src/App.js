import "./App.css";
import Header from "./components/Header";
import Plot from "./components/Plot";
import { io } from "socket.io-client";
import { useState } from "react";

function App() {
  const socket = io("/");
  const [data, setData] = useState([]);

  socket.on("data", (new_data) => {
    setData([...data, new_data]);
  });

  return (
    <div className="App">
      <Header />
      <Plot title="Learning Rate" data={data} y_key="lr" x_key="epoch" />
      <Plot
        title="Train Accuracy"
        data={data}
        y_key="train_acc"
        x_key="epoch"
      />
      <Plot title="Test Accuracy" data={data} y_key="test_acc" x_key="epoch" />
      <Plot title="Train Loss" data={data} y_key="train_loss" x_key="epoch" />
      <Plot title="Test Loss" data={data} y_key="test_loss" x_key="epoch" />
    </div>
  );
}

export default App;
