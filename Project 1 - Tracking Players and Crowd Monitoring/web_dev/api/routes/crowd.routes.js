import express from 'express';
import {Crowd} from '../models/crowd.model.js';

const router = express.Router();

router.get('/', async (req, res) => {
  try {
    const crowdData = await Crowd.find();
    res.status(200).json(crowdData);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

router.get('/heat-map', async (req, res) => {
  try {
    const crowdHeatMapData = await Crowd.find().select('heatMapData');
    res.status(200).json(crowdHeatMapData);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

router.get('/eventsOrIncidents', async (req, res) => {
  try {
    // Assuming events or incidents data is stored as a string in the database
    const crowdEventsOrIncidents = await Crowd.find().select('eventsOrIncidents');
    res.status(200).json(crowdEventsOrIncidents);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

router.get('/trends', async (req, res) => {
  try {
    const crowdTrends = await Crowd.find().select('trends');
    res.status(200).json(crowdTrends);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

router.post('/', async (req, res) => {
    try {
        if (
            !req.body.timestamp ||
            !req.body.location ||
            !req.body.density ||
            !req.body.heatMapData ||
            !req.body.eventsOrIncidents ||
            !req.body.trends
        ) {
            return res.status(400).send({ message: "Missing parameters" });
        }
      const CrowdData = {
        timestamp: req.body.timestamp,
        location: req.body.location,
        density: req.body.density,
        heatMapData: req.body.heatMapData,
        eventsOrIncidents: req.body.eventsOrIncidents,
        trends: req.body.trends,
      };

      const newCrowdData = await Crowd.create(CrowdData);
  
      res.status(201).json(newCrowdData);
    } catch (error) {
      res.status(500).json({ message: error.message });
    }
  });

router.post('/heat-map', async (req, res) => {
try {
    if (!req.body.location || !req.body.heatMapData) {
    return res.status(400).send({ message: 'Missing parameters (enter heat map data and location)' });
    }

    const crowdRecord = await Crowd.findOne({ location: req.body.location });

    if (!crowdRecord) {
    return res.status(404).json({ message: 'Crowd record not found' });
    }


    crowdRecord.heatMapData = req.body.heatMapData;


    await crowdRecord.save();

    res.status(200).json(crowdRecord);
} catch (error) {
    res.status(500).json({ message: error.message });
}
});

router.post('/eventsOrIncidents', async (req, res) => {
  try {
    if (!req.body.location || !req.body.eventsOrIncidents) {
      return res.status(400).send({ message: 'Missing parameters' });
    }

    const crowdRecord = await Crowd.findOne({ location: req.body.location });

    if (!crowdRecord) {
      return res.status(404).json({ message: 'Crowd record not found' });
    }


    crowdRecord.eventsOrIncidents = req.body.eventsOrIncidents;


    await crowdRecord.save();

    res.status(200).json(crowdRecord);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

router.post('/trends', async (req, res) => {
  try {
    if (!req.body.location || !req.body.trends) {
      return res.status(400).send({ message: 'Missing parameters' });
    }


    const crowdRecord = await Crowd.findOne({ location: req.body.location });

    if (!crowdRecord) {
      return res.status(404).json({ message: 'Crowd record not found' });
    }


    crowdRecord.trends = req.body.trends;


    await crowdRecord.save();

    res.status(200).json(crowdRecord);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const updatedData = req.body;
    if (
      !updatedData.timestamp ||
      !updatedData.location ||
      !updatedData.density ||
      !updatedData.heatMapData ||
      !updatedData.eventsOrIncidents ||
      !updatedData.trends
    ) {
      return res.status(400).send({ message: 'Missing parameters' });
    }

    const crowdRecord = await Crowd.findById(id);

    if (!crowdRecord) {
      return res.status(404).json({ message: 'Crowd record not found' });
    }

    const result = await Object.assign(crowdRecord, updatedData);

    if (!result) {
      return res.status(500).json({ message: 'Error updating crowd record' });
    }

    await crowdRecord.save(); // Save the updated crowd record

    res.status(200).json({ message: 'Data updated successfully' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

router.put('/heat-map/:id', async (req, res) => {
  try {
    const crowdId = req.params.id;
    const updatedHeatMapData = req.body.heatMapData;

    if (!req.body.heatMapData) {
      return res.status(400).send({ message: 'Missing parameters' });
    }


    const crowdRecord = await Crowd.findById(crowdId);

    if (!crowdRecord) {
      return res.status(404).json({ message: 'Crowd record not found' });
    }


    crowdRecord.heatMapData = updatedHeatMapData;


    await crowdRecord.save();

    res.status(200).json(crowdRecord);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});


router.put('/eventsOrIncidents/:id', async (req, res) => {
  try {
    const crowdId = req.params.id;
    const updatedEventsOrIncidents = req.body.eventsOrIncidents;

    if (!req.body.eventsOrIncidents) {
      return res.status(400).send({ message: 'Missing parameters' });
    }
    const crowdRecord = await Crowd.findById(crowdId);

    if (!crowdRecord) {
      return res.status(404).json({ message: 'Crowd record not found' });
    }


    crowdRecord.eventsOrIncidents = updatedEventsOrIncidents;


    await crowdRecord.save();

    res.status(200).json(crowdRecord);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});


router.put('/trends/:id', async (req, res) => {
  try {
    const crowdId = req.params.id;
    const updatedTrends = req.body.trends;


    const crowdRecord = await Crowd.findById(crowdId);

    if (!crowdRecord) {
      return res.status(404).json({ message: 'Crowd record not found' });
    }


    crowdRecord.trends = updatedTrends;


    await crowdRecord.save();

    res.status(200).json(crowdRecord);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

export default router;
