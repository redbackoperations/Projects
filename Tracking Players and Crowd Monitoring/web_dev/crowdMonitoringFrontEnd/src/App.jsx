import React from 'react';
import { Routes, Route } from 'react-router-dom';
import CreatePlayer from './Pages/createPlayer.jsx';
import UpdatePlayer from './Pages/UpdatePlayer.jsx';
import UpdatePlayerStats from './Pages/UpdatePlayerStats.jsx';
import DeletePlayer from './Pages/DeletePlayer.jsx';
import ListPlayers from './Pages/ListPlayer.jsx';
import PlayerMovement from './Pages/PlayerMovement.jsx';
import PlayerDetails from './Pages/PlayerDetails.jsx';
import Crowd from './Pages/Crowd/Crowd.jsx';
import CrowdEvents from './Pages/Crowd/CrowdEvents.jsx';
import CrowdHM from './Pages/Crowd/CrowdHM.jsx';
import CrowdsTrends from './Pages/Crowd/CrowdsTrends.jsx';
import { elements } from 'chart.js';
import  Navbar  from "../src/components/Navbar.jsx";
import  Features  from "./components/Features.jsx";
import  Description  from "./components/Description.jsx";
import  Home  from "./Pages/Home/Home.jsx";
import  Login  from "./Pages/Home/Login.jsx";
import  Signup  from "./Pages/Home/Signup.jsx";


const App = () => {
  return (
    <Routes>
      <Route path='/' element={<Home/>} />
      <Route path='/player/create' element={<CreatePlayer />} />
      <Route path='/player/update/:id' element={<UpdatePlayer />} />
      <Route path='/player/update/stats/:id' element={<UpdatePlayerStats />} />
      <Route path='/player/delete/:id' element={<DeletePlayer />} />
      <Route path='/players' element={<ListPlayers />} />
      <Route path='/player/movements/:id' element={<PlayerMovement />} />
      <Route path='/player/:id' element={<PlayerDetails />} />
      <Route path='/crowd' element={<Crowd/>} />
      <Route path='/crowd/events' element={<CrowdEvents/>} />
      <Route path='/crowd/trends' element={<CrowdsTrends/>} />
      <Route path='/crowd/heatmap' element={<CrowdHM/>} />
    </Routes>
  );
};

export default App;
