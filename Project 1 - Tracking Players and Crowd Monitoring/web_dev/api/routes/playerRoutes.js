import express from 'express';

import {HeartRate} from '../models/heart.model.js';
import {Player} from '../models/players.model.js';
const router = express.Router();



router.post('/', async (req, res) => {
    try {
        if (
            !req.body.firstName ||
            !req.body.lastName ||
            !req.body.age ||
            !req.body.team ||
            !req.body.position ||
            !req.body.dateOfBirth ||
            !req.body.sport
        ) {
            return res.status(400).send({ message: "Missing parameters" });
        }

        const newPlayer = {
            firstName: req.body.firstName,
            lastName: req.body.lastName,
            age: req.body.age,
            team: req.body.team,
            position: req.body.position,
            dateOfBirth: req.body.dateOfBirth,
            sport: req.body.sport,
            heartRate: req.body.heartRate || null,
            movements: req.body.movements || [], 
            statistics: req.body.statistics || {},
        };

        const player = await Player.create(newPlayer);
        return res.status(201).send(player);
    } catch (err) {
        console.error(err);
        res.status(500).send(err.message);
    }
});


router.get('/', async (req, res) =>
{
    try {
        const players = await Player.find({});
        return res.status(200).json({count : players.length, data : players});
    } catch (error) {
        console.log(error);
        res.send(500).json({message : error.message});
    }
}
);

router.get('/:id/statistics', async (req, res) => {
    try {
      const { id } = req.params;
      const player = await Player.findById(id);
  
      if (!player) {
        return res.status(404).json({ message: 'Player not found' });
      }
  
      const playerStatistics = player.statistics; // Assuming 'statistics' is a field in the player document
  
      res.status(200).json(playerStatistics);
    } catch (error) {
      res.status(500).json({ message: error.message });
    }
  });

  router.put('/:id/statistics', async (req, res) => {
    const { id } = req.params;
    const {
      gamesPlayed,
      goalsScored,
      assists,
      yellowCards,
      redCards,
    } = req.body;
  
    try {
      const player = await Player.findById(id);
  
      if (!player) {
        return res.status(404).json({ message: 'Player not found' });
      }
  
      // Validate input data
      if (typeof gamesPlayed !== 'number' || gamesPlayed < 0 ||
          typeof goalsScored !== 'number' || goalsScored < 0 ||
          typeof assists !== 'number' || assists < 0 ||
          typeof yellowCards !== 'number' || yellowCards < 0 ||
          typeof redCards !== 'number' || redCards < 0) {
        return res.status(400).json({ message: 'Invalid input data' });
      }

      // Update player's statistics
      player.statistics.gamesPlayed = gamesPlayed;
      player.statistics.goalsScored = goalsScored;
      player.statistics.assists = assists;
      player.statistics.yellowCards = yellowCards;
      player.statistics.redCards = redCards;
  
      await player.save();
  
      // Return the updated player object with statistics
      res.status(200).json(player);
    } catch (error) {
      res.status(500).json({ message: error.message });
    }
});

router.get('/:id/movements', async (req, res) => {
    try {
        const { id } = req.params;
        
        if (id.length !== 24) {
            return res.status(400).json({ message: 'Invalid player id' });
        }

        const player = await Player.findById(id).select('movements');

        if (!player) {
            return res.status(404).json({ message: 'Player not found' });
        }

        return res.status(200).json({ movements: player.movements });
    } catch (error) {
        console.log(error);
        res.status(500).json({ message: error.message });
    }
});

router.post('/:id/movements', async (req, res) => {
    try {
        if (
            !req.body.time ||
            !req.body.date ||
            !req.body.location ||
            !req.body.alt ||
            !req.body.sat
        ) {
            return res.status(400).send({ message: "Missing parameters" });
        }

        const { id } = req.params;
        
        if (id.length !== 24) {
            return res.status(400).json({ message: 'Invalid player id' });
        }

        const player = await Player.findById(id);

        if (!player) {
            return res.status(404).json({ message: 'Player not found' });
        }


        const { time, date, location, alt, sat } = req.body;

        const newMovement = {
            time,
            date,
            location,
            alt,
            sat,
        };

        player.movements.push(newMovement);

        await player.save();

        return res.status(201).json({ message: 'Movement data added successfully', movement: newMovement });
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: error.message });
    }
});

router.get('/:id', async (req, res) =>
{
    try {
        const {id} = req.params;
        if (id.length != 24) {
            return res.status(400).json({ message: 'Invalid player id' });
        }
        const player = await Player.findById(id);
        if (!player) {
            return res.status(404).send({ message: "Player not found" });
        }
        return res.status(200).json(player);
    } catch (error) {
        console.log(error);
        res.send(500).json({message : error.message});
    }
}
);

router.delete('/:id', async (req, res) =>
{
    try {
        const {id} = req.params;
        if (id.length != 24) {
            return res.status(400).json({ message: 'Invalid player id' });
        }
        const player = await Player.findByIdAndDelete(id);
        if (!player) {
            return res.status(404).send({ message: "Player not found" });
        }
        return res.status(200).json(player);
    } catch (error) {
        console.log(error);
        res.send(500).json({message : error.message});
    }
}
);

router.put('/:id', async (req, res) => {
    try {
        if (
            !req.body.firstName ||
            !req.body.lastName ||
            !req.body.age ||
            !req.body.team ||
            !req.body.position ||
            !req.body.dateOfBirth ||
            !req.body.sport
        ) {
            return res.status(400).send({ message: "Missing parameters" });
        }
        const {id} = req.params;

        const result = await Player.findByIdAndUpdate(id , req.body);

        if (!result) 
        {
            return res.status(404).json({message : 'player not found'});
        }
        return res.status(200).json({message : 'player updated successfully'}); 
        }
    catch (err) {
        console.log(err);
        res.status(500).json({message : err.message});
    }
});

export default router;

