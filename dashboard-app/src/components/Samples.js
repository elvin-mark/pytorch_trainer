import React from "react";
import Sample from "./Sample";
import "./Trainer.css";

function Samples() {
  const rand = [
    {
      img: null,
      data: [
        { class: "test1", prob: 28 },
        { class: "test2", prob: 90 },
        { class: "test3", prob: 28 },
        { class: "test4", prob: 90 },
        { class: "test5", prob: 28 },
      ],
    },
    {
      img: null,
      data: [
        { class: "test1", prob: 28 },
        { class: "test2", prob: 90 },
        { class: "test3", prob: 28 },
        { class: "test4", prob: 90 },
        { class: "test5", prob: 28 },
      ],
    },
    {
      img: null,
      data: [
        { class: "test1", prob: 28 },
        { class: "test2", prob: 90 },
        { class: "test3", prob: 28 },
        { class: "test4", prob: 90 },
        { class: "test5", prob: 28 },
      ],
    },
  ];

  return (
    <div>
      <h1 className="Title">Some Samples</h1>
      <div>
        {rand.map((elem) => (
          <Sample data={elem.data}></Sample>
        ))}
      </div>
    </div>
  );
}

export default Samples;
