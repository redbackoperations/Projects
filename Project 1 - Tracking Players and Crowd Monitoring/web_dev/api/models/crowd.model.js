import mongoose from "mongoose";

const crowdSchema = new mongoose.Schema({
  timestamp: {
    type: Date,
    required: true,
  },
  location: {
    type: String,
    required: true,
  },
  density: {
    type: Number,
    required: true,
  },
  heatMapData: {
    type: Object, 
  },
  eventsOrIncidents: {
    type: String,
    required: false, 
  },
  trends: {
    type: String,
    required: false,
}
  
});

export const Crowd = mongoose.model('Crowd', crowdSchema);
