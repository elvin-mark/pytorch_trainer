import React from "react";
import Sample from "./Sample";
import "./Trainer.css";

function Samples({ data }) {
  return (
    <div>
      <h1 className="Title">Some Samples</h1>
      <div>
        {data.map((elem) => (
          <Sample data={elem.data} img={elem.img}></Sample>
        ))}
      </div>
    </div>
  );
}

export default Samples;
