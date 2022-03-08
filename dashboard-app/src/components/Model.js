import React from "react";

function ModelGraph({ graph }) {
  return (
    <div>
      <h1 className="Title">Model Information</h1>
      <h3>Model Graph</h3>
      <img src={graph}></img>
    </div>
  );
}

export default ModelGraph;
