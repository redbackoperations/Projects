import React from "react";

const Features = () => {
  return (
    <div>
      <h4 className="card-title mt-5 mb-5">Features</h4>
      <div className="row ">
      <div className="col-md-3">
        <div className="card" >
          <img
            src="../../src/assets/image1.jpeg"
            className="card-img-top"
            alt="Image"
          />
          <div className="card-body">
            <h4 className="card-title">Data Visualization</h4>
            <p className="card-text">
            We provide a comprehensive understanding of individuals and crowds through efficient visualization of data to provide a comprehensive understanding of the environment. 
            </p>
          </div>
        </div>
        </div>
        <div className="col-md-3">
        <div className="card" >
          <img
            src="../../src/assets/image2.jpeg"
            className="card-img-top"
            alt="Image"
          />
          <div className="card-body">
            <h4 className="card-title">Crowd Density Tracking</h4>
            <p className="card-text">
            Track crowd density with our overcrowding detection via device clustering to provide real time alerts to keep population dense venues safe. 
            </p>
          </div>
        </div>
        </div>
        <div className="col-md-3">
        <div className="card" >
          <img
            src="../../src/assets/image3.jpeg"
            className="card-img-top"
            alt="Image"
          />
          <div className="card-body">
            <h4 className="card-title">User Tracking</h4>
            <p className="card-text">
            Track individuals across frames, taking into account heartrate, location and activity recognition compounded with a cutting-edge approach to individual collision prediction. 
            </p>
          </div>
        </div>
        </div>
        <div className="col-md-3">
        <div className="card" >
          <img
            src="../../src/assets/image4.jpeg"
            className="card-img-top"
            alt="Image"
          />
          <div className="card-body">
            <h4 className="card-title">Heatmap</h4>
            <p className="card-text">
              SOvercrowding can be detected and prevented through our precise heatmaps which are swiftly generated from GPS data. 
            </p>
          </div>
        </div>
        </div>
       
      </div>
    </div>
  );
};

export default Features;