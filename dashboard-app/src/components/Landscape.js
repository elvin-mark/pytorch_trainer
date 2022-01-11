import React from "react";

function Landscape({ surf, contour }) {
  return (
    <div>
      <h1 className="Title">Landscape</h1>
      <h3>Landscape Surface</h3>
      <img src={surf}></img>
      <h3>Landscape Contour</h3>
      <img src={contour}></img>
    </div>
  );
}

export default Landscape;
