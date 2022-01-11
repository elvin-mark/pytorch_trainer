import { rgbToHex } from "@mui/material";
import React from "react";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";

function Sample({ img, data }) {
  return (
    <div>
      <table>
        <tr>
          <td>
            <img src={img}></img>
          </td>
          <td>
            <BarChart
              width={500}
              height={400}
              data={data}
              layout="vertical"
              margin={{ left: 10 }}
            >
              <Bar dataKey="prob" fill={rgbToHex("#0077CC")}></Bar>
              <CartesianGrid stroke="#ccc"></CartesianGrid>
              <XAxis type="number"></XAxis>
              <YAxis dataKey="class" type="category"></YAxis>
            </BarChart>
          </td>
        </tr>
      </table>
    </div>
  );
}

export default Sample;
