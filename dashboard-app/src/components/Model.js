import React from "react";
import Graph from "react-vis-network-graph";
function Model({ graph_img, graph_struct }) {
  const options = {
    layout: {
      improvedLayout: true,
      clusterThreshold: true,
    },
    edges: {
      color: "#000000",
    },
    height: "500px",
  };

  const events = {
    select: function (event) {
      var { nodes, edges } = event;
    },
  };
  return (
    <div>
      <h1 className="Title">Model Information</h1>
      <h3>Model Graph</h3>
      <Graph
        graph={graph_struct}
        options={options}
        events={events}
        getNetwork={(network) => {}}
      ></Graph>
      <img src={graph_img}></img>
    </div>
  );
}

export default Model;
