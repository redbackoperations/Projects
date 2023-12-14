import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';

class MapSample extends StatefulWidget {
  @override
  State<MapSample> createState() => MapSampleState();
}

class MapSampleState extends State<MapSample> {
  final Completer<GoogleMapController> _controller =
  Completer<GoogleMapController>();

  static const CameraPosition _kBurwood = CameraPosition(
    target: LatLng(-33.879190, 151.103818),
    zoom: 15.0,
  );

  static const LatLng _destinationLatLng = LatLng(-33.880190, 151.102818);

  Set<Marker> _markers = {
    Marker(
      markerId: MarkerId('destination_marker'),
      position: _destinationLatLng,
      infoWindow: InfoWindow(
      ),
    ),
  };

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GoogleMap(
        mapType: MapType.normal,
        initialCameraPosition: _kBurwood,
        markers: _markers,
        onMapCreated: (GoogleMapController controller) {
          _controller.complete(controller);
        },
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _goToTheLake,
        label: const Text('Go to Destination'),
        icon: const Icon(Icons.directions),
      ),
    );
  }

  Future<void> _goToTheLake() async {
    final GoogleMapController controller = await _controller.future;
    await controller.animateCamera(CameraUpdate.newLatLng(_destinationLatLng));
  }
}
