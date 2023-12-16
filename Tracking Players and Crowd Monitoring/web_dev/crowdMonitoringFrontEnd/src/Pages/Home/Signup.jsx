import React from "react";
import { Link } from "react-router-dom";

const Signup = () => {
  return (
    <>
      <center>
        <div
          className="card mt-5"
          style={{
            width: "24rem",
          }}
        >
          <div className="card-body">
            <div className="card-header  border-bottom-0">
              <h1 className="fw-bold mb-0 fs-2">Signup</h1>
            </div>
            <div className="card-body ">
              <form className="">
                <div className="form-floating mb-3">
                  <input
                    type="text"
                    className="form-control rounded-3"
                    id="floatingInput"
                    placeholder="Username"
                    style={{
                      backgroundImage: 'url("data:image/png',
                      backgroundRepeat: "no-repeat",
                      backgroundAttachment: "scroll",
                      backgroundSize: "16px 18px",
                      backgroundPosition: "98% 50%",
                    }}
                  />
                  <label htmlFor="floatingInput">Username</label>
                </div>
                <div className="form-floating mb-3">
                  <input
                    type="email"
                    className="form-control rounded-3"
                    id="floatingInput"
                    placeholder="name@example.com"
                    style={{
                      backgroundImage: 'url("data:image/png',
                      backgroundRepeat: "no-repeat",
                      backgroundAttachment: "scroll",
                      backgroundSize: "16px 18px",
                      backgroundPosition: "98% 50%",
                    }}
                  />
                  <label htmlFor="floatingInput">Email</label>
                </div>
                <div className="form-floating mb-3">
                  <input
                    type="text"
                    className="form-control rounded-3"
                    id="floatingInput"
                    placeholder="999999999"
                    style={{
                      backgroundImage: 'url("data:image/png',
                      backgroundRepeat: "no-repeat",
                      backgroundAttachment: "scroll",
                      backgroundSize: "16px 18px",
                      backgroundPosition: "98% 50%",
                    }}
                  />
                  <label htmlFor="floatingInput">Phone number</label>
                </div>
                <div className="form-floating mb-3">
                  <input
                    type="password"
                    className="form-control rounded-3"
                    id="floatingPassword"
                    placeholder="Password"
                    style={{
                      backgroundImage: 'url("data:image/png',
                      backgroundRepeat: "no-repeat",
                      backgroundAttachment: "scroll",
                      backgroundSize: "16px 18px",
                      backgroundPosition: "98% 50%",
                    }}
                  />
                  <label htmlFor="floatingPassword">Password</label>
                </div>
                <button
                  className="w-100 mb-2 btn btn-lg rounded-5 btn-primary"
                  type="submit"
                >
                  SIGNUP
                </button>
                <hr className="my-4" />
                <h2 className="fs-5 fw-bold mb-3">Or Login Using</h2>
                <Link className="text-body-secondary" to={"/login"}>
                  Login
                </Link>
              </form>
            </div>
          </div>
        </div>
      </center>
    </>
  );
};

export default Signup;