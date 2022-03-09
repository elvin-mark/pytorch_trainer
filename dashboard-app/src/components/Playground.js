import React from "react";
import {
  PlaygroundInputImage,
  PlaygroundInputSketch,
  PlaygroundOutputLabels,
} from "react-dl-playground";

function Playground({ onSubmitSample, response }) {
  console.log(response);
  return (
    <div>
      <h1 className="Title">Playground</h1>
      <h3 style={{ textAlign: "left" }}>Uploading an Image</h3>
      <div
        style={{
          display: "flex",
          alignItems: "center",
        }}
      >
        <PlaygroundInputImage onSubmit={onSubmitSample}></PlaygroundInputImage>
        <PlaygroundOutputLabels {...response}></PlaygroundOutputLabels>
      </div>
      <h3 style={{ textAlign: "left" }}>Draw yourself!</h3>
      <div style={{ display: "flex", alignItems: "center" }}>
        <PlaygroundInputSketch
          onSubmit={onSubmitSample}
          width="280px"
          height="280px"
        ></PlaygroundInputSketch>
        <PlaygroundOutputLabels {...response}></PlaygroundOutputLabels>
      </div>
    </div>
  );
}

export default Playground;
