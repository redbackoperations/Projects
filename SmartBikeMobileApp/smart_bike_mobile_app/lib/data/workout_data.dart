class WorkoutData {
  final DateTime timestamp;
  final num duration, totalPwr, totalRPM, totalDst;

  const WorkoutData({
    required this.timestamp,
    required this.duration,
    required this.totalPwr,
    required this.totalRPM,
    required this.totalDst
  });

  toJson() {
    return {
      'timeStamp': timestamp,
      'duration': duration,
      'totalPower': totalPwr,
      'totalRPM': totalRPM,
      'distance': totalDst
    };
  }
}