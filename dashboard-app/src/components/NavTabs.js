import { Tab, Tabs } from "@mui/material";
import React from "react";
import { Link } from "react-router-dom";

function NavTabs() {
  return (
    <div>
      <Tabs>
        <Tab label="Training" value="/" to="/" component={Link}></Tab>
        <Tab
          label="Samples"
          value="/samples"
          to="/samples"
          component={Link}
        ></Tab>
        <Tab
          label="Landscape"
          value="/Landscape"
          to="/Landscape"
          component={Link}
        ></Tab>
        <Tab label="Model" value="/model" to="/model" component={Link}></Tab>
      </Tabs>
    </div>
  );
}

export default NavTabs;
