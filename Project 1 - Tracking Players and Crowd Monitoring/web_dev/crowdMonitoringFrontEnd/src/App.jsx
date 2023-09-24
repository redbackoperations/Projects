import React from 'react';
import { Routes, Route } from 'react-router-dom';
import CreatePlayer from './Pages/createPlayer.jsx';
import UpdatePlayer from './Pages/UpdatePlayer.jsx';
import UpdatePlayerStats from './Pages/UpdatePlayerStats.jsx';
import DeletePlayer from './Pages/DeletePlayer.jsx';
import ListPlayers from './Pages/ListPlayer.jsx';
import PlayerMovement from './Pages/PlayerMovement.jsx';
import PlayerDetails from './Pages/PlayerDetails.jsx';

const App = () => {
  return (
    <Routes>
      <Route path='/player/create' element={<CreatePlayer />} />
      <Route path='/player/update/:id' element={<UpdatePlayer />} />
      <Route path='/player/update/stats/:id' element={<UpdatePlayerStats />} />
      <Route path='/player/delete/:id' element={<DeletePlayer />} />
      <Route path='/players' element={<ListPlayers />} />
      <Route path='/player/movements/:id' element={<PlayerMovement />} />
      <Route path='/player/:id' element={<PlayerDetails />} />
    </Routes>
  );
};

export default App;
