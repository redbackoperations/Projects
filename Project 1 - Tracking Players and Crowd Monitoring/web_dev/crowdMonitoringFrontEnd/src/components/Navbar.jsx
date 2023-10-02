import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav
      className="navbar bg-dark border-bottom border-body navbar-expand-lg bg-body-tertiary"
      data-bs-theme="dark"
    >
      <div className="container-fluid">
        <a className="navbar-brand" href="#">
            Project
        </a>
        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarText"
          aria-controls="navbarText"
          aria-expanded="false"
          aria-label="Toggle navigation"
        >
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarText">
          <ul className="navbar-nav me-auto mb-2 mb-lg-0">
            <li className="nav-item">
              <Link className="nav-link" aria-current="page" to="/">
                Home
              </Link>
            </li>
            <li className="nav-item">
              <a className="nav-link" href="#">
                Heatmap
              </a>
            </li>
            <li className="nav-item">
              <a className="nav-link" href="#">
                Crowd Density
              </a>
            </li>
            <li className="nav-item">
              <a className="nav-link" href="#">
                Data Visualisation
              </a>
            </li>
            <li className="nav-item">
              <a className="nav-link" href="#">
                User Tracking
              </a>
            </li>
            <li className="nav-item">
              <a className="nav-link" href="#">
                About
              </a>
            </li>
          </ul>
          <Link to={'/'} className="nav-link" ><span className="nav-link" >Login</span></Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;