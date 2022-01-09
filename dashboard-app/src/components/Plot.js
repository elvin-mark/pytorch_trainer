import React from "react";

import {
  LineChart,
  Tooltip,
  XAxis,
  Line,
  CartesianGrid,
  YAxis,
} from "recharts";

function Plot({ title, data, x_key, y_key }) {
  return (
    <div style={{ float: "left" }}>
      <h3 style={{ textAlign: "center" }}>{title}</h3>
      <LineChart
        width={400}
        height={400}
        data={data}
        margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
      >
        <XAxis dataKey={x_key} />
        <Tooltip />
        <CartesianGrid stroke="#f5f5f5" />
        <YAxis dataKey={y_key}></YAxis>
        <Line type="monotone" dataKey={y_key} stroke="#ff7300" />
      </LineChart>
    </div>
  );
}

export default Plot;
