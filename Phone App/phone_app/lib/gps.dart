import 'dart:async';

import 'package:flutter/material.dart';
import 'package:location/location.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';

class MapPage extends StatefulWidget {
  MapPage({super.key});
  bool _isTracking = false;
  void startStopTracking(bool isTracking) {
    _isTracking = isTracking;
  }

  @override
  State<MapPage> createState() => _MapPageState();
}

class _MapPageState extends State<MapPage> {
  final Completer<GoogleMapController> _mapController =
      Completer<GoogleMapController>();
  Location _locationController = new Location();
  LatLng? _currentPosition = null;
  bool _isTracking = false;

  @override
  void initState() {
    super.initState();
    setLocation();
    //Timer.periodic(Duration(seconds: 1), (timer) {
    //getLocationUpdates();
    //});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _currentPosition == null
          ? Center(
              child: Text("Loading..."),
            )
          : GoogleMap(
              onMapCreated: ((GoogleMapController controller) =>
                  _mapController.complete(controller)),
              initialCameraPosition: CameraPosition(
                target: _currentPosition!,
                zoom: 15,
              ),
              markers: {
                // Marker(
                //     markerId: MarkerId("_startLocation"),
                //     icon: BitmapDescriptor.defaultMarker,
                //     position: _currentPosition!),
                Marker(
                    markerId: MarkerId("_currentLocation"),
                    icon: BitmapDescriptor.defaultMarker,
                    position: _currentPosition!)
              },
              //TODO Make start pos set when user clicks begi workout.
            ),
    );
  }

  Future<void> cameraFollow(LatLng pos) async {
    GoogleMapController controller = await _mapController.future;
    CameraPosition newCameraPosition = CameraPosition(
      target: pos,
      zoom: 15,
    );
    await controller.animateCamera(
      CameraUpdate.newCameraPosition(newCameraPosition),
    );
  }

  Future<void> setLocation() async {
    // Begin checking location services available and user has permission
    bool serviceEnabled;
    PermissionStatus permissionGranted;

    serviceEnabled = await _locationController.serviceEnabled();
    if (serviceEnabled) {
      serviceEnabled = await _locationController.requestService();
    } else {
      return;
    }

    permissionGranted = await _locationController.hasPermission();
    // end permission and service checks
    if (permissionGranted == PermissionStatus.denied) {
      permissionGranted = await _locationController.requestPermission();

      if (permissionGranted != PermissionStatus.granted) {
        return;
      }
    }

    // _locationController.onLocationChanged
    //     .listen((LocationData currentLocation) {

    LocationData currentLocation = await _locationController.getLocation();
    if (currentLocation.latitude != null && currentLocation.longitude != null) {
      setState(() {
        _currentPosition =
            LatLng(currentLocation.latitude!, currentLocation.longitude!);
      });
      cameraFollow(_currentPosition!);
    }

    //});
  } // end getLocationUpdates

  Future<void> getLocationUpdates() async {
    // Begin checking location services available and user has permission
    bool serviceEnabled;
    PermissionStatus permissionGranted;

    serviceEnabled = await _locationController.serviceEnabled();
    if (serviceEnabled) {
      serviceEnabled = await _locationController.requestService();
    } else {
      return;
    }

    permissionGranted = await _locationController.hasPermission();
    // end permission and service checks
    if (permissionGranted == PermissionStatus.denied) {
      permissionGranted = await _locationController.requestPermission();

      if (permissionGranted != PermissionStatus.granted) {
        return;
      }
    }

    // _locationController.onLocationChanged
    //     .listen((LocationData currentLocation) {
    if (_isTracking) {
      LocationData currentLocation = await _locationController.getLocation();
      if (currentLocation.latitude != null &&
          currentLocation.longitude != null) {
        setState(() {
          _currentPosition =
              LatLng(currentLocation.latitude!, currentLocation.longitude!);
        });
        cameraFollow(_currentPosition!);
      }
    }
    //});
  } // end getLocationUpdates
}// end class


