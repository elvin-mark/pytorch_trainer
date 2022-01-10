import React from "react";
import Plot from "./Plot";
import "./Trainer.css";

function TrainingPlots({ data }) {
  return (
    <div>
      <h1 className="Title">Training Dashboard</h1>
      <div>
        <Plot title="Learning Rate" data={data} y_key="lr" x_key="epoch" />
        <Plot
          title="Train Accuracy"
          data={data}
          y_key="train_acc"
          x_key="epoch"
        />
        <Plot
          title="Test Accuracy"
          data={data}
          y_key="test_acc"
          x_key="epoch"
        />
        <Plot title="Train Loss" data={data} y_key="train_loss" x_key="epoch" />
        <Plot title="Test Loss" data={data} y_key="test_loss" x_key="epoch" />
      </div>
    </div>
  );
}

export default TrainingPlots;
