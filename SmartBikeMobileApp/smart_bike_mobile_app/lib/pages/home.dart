import 'dart:async';
import 'dart:convert';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';
import 'package:smart_bike_mobile_app/data/workout_data.dart';
import 'package:smart_bike_mobile_app/pages/workout_record.dart';
import 'package:smart_bike_mobile_app/utils/firebase_auth_util.dart';
import 'package:smart_bike_mobile_app/pages/edit_profile.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import 'login.dart';

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  // Timer vars
  int numH = 0, numM = 0, numS = 0;
  String strH = '00', strM = '00', strS = '00';
  Timer? timer;
  bool isRunning = false;

  // Workout stat vars
  num totalRPM = 0, totalPwr = 0, duration = 0;

  // Page Layout vars
  final bikeIdController = TextEditingController();
  int screenId = 0;
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  // Firebase Authentication
  final firebaseAuthUtil = FirebaseAuthUtil();

  // Cloud Firestore Database
  FirebaseFirestore db = FirebaseFirestore.instance;
  late int recordSize;
  late Iterable<Object?> snapshotData;

  Future<void> addRecord() {
    String uid = firebaseAuthUtil.auth.currentUser!.uid;
    return db.collection(uid).add(WorkoutData(
        timestamp: DateTime.timestamp(),
        duration: duration,
        totalPwr: (totalPwr * 10).round() / 10,
        totalRPM: (totalRPM * 10).round() / 10,
        totalDst: (totalRPM * 100).round() / 100).toJson())
    .then((val) => debugPrint('Added: $val'))
    .catchError((e) => debugPrint('Error: $e'));
  }

  Future<List> getRecord() async {
    String uid = firebaseAuthUtil.auth.currentUser!.uid;
    QuerySnapshot snapshot = await db.collection(uid).get();
    Iterable<Object?> snapshotData = snapshot.docs.map((doc) => doc.data());
    return [snapshotData, snapshot.size];
  }

  // MQTT
  bool mqttConnected = false;
  MqttServerClient? mqttServerClient;

  // Bike data
  num power = 10.0, speed = 10.0, rpm = 10;
  num distance = 0.0;
  num resistance = 0.0, incline = 0;
  String bikeId = '';

  Future<void> connectMqtt() async {
    await dotenv.load(fileName: '.env');

    // Set up MQTT Login Credentials with dotenv
    String mqttHost = dotenv.get('MQTT_TEST_HOST');
    String mqttUser = dotenv.get('MQTT_TEST_USER');
    String mqttPass = dotenv.get('MQTT_TEST_PASS');
    String mqttClid = dotenv.get('MQTT_CLID');

    MqttServerClient client =
    MqttServerClient.withPort(mqttHost, mqttClid, 8883);
    client.secure = true;

    client.logging(on: true);
    client.onConnected = () {
      debugPrint('Connected');
      setState(() {
        mqttConnected = true;
      });
    };
    client.onDisconnected = () {
      debugPrint('Disconnected');
      setState(() {
        mqttConnected = false;
        bikeId = '';
      });
    };
    client.onSubscribed = (topic) {
      debugPrint('Sub to topic: $topic');
    };
    client.onSubscribeFail = (topic) {
      debugPrint('Failed to sub to topic: $topic');
    };
    client.onUnsubscribed = (topic) {
      debugPrint('Unsubed from topic: $topic');
    };

    final connMess =
    MqttConnectMessage().authenticateAs(mqttUser, mqttPass).startClean();

    client.connectionMessage = connMess;

    try {
      await client.connect(mqttUser, mqttPass);
    } catch (e) {
      debugPrint('MQTT Connection ERROR: $e');
      client.disconnect();
      return;
    }

    var topic = 'bike/$bikeId/#';
    client.subscribe(topic, MqttQos.atMostOnce);

    client.updates?.listen((List<MqttReceivedMessage<MqttMessage?>>? c) {
      final recMess = c![0].payload as MqttPublishMessage;
      final pt =
      MqttPublishPayload.bytesToStringAsString(recMess.payload.message);
      handleMessage(c[0].topic, pt);
    });

    mqttServerClient = client;
  }

  void handleMessage(String topic, String pt) {
    debugPrint("TOPIC: $topic");
    var topicComponents = topic.split("/");
    debugPrint("DEBUG: ${topicComponents.last}");
    debugPrint("DEBUG: $pt");

    switch (topicComponents.last) {
      case "speed":
        var value = json.decode(pt)['value'];
        setState(() {
          speed = value;
        });
        debugPrint("Message from topic $topic received: $pt");
        break;
      case "cadence":
        var value = json.decode(pt)['value'];
        setState(() {
          rpm = value;
        });
        debugPrint("Message from topic $topic received: $pt");
        break;
      case "power":
        var value = json.decode(pt)['value'];
        setState(() {
          power = value;
        });
        debugPrint("Message from topic $topic received: $pt");
        break;
      case "resistance":
        var value = json.decode(pt);
        setState(() {
          resistance = value;
        });
        debugPrint("Message from topic $topic received: $pt");
        break;
      case "incline":
        var value = json.decode(pt);
        setState(() {
          incline = value;
        });
        debugPrint("Message from topic $topic received: $pt");
        break;
      default:
        debugPrint("Unknown topic $topic encountered");
    }
  }

  void publishMqtt(String topic, num val) {
    var builder = MqttClientPayloadBuilder();
    String msg = '{"value": ${val.toStringAsFixed(1)}}';
    builder.addString(msg);
    mqttServerClient!
        .publishMessage(topic, MqttQos.atMostOnce, builder.payload!);
  }

  Future<void> disconnectMqtt() async {
    mqttServerClient!.unsubscribe('bike/$bikeId/#');
    await MqttUtilities.asyncSleep(2);
    debugPrint('Disconnecting from server...');
    mqttServerClient!.disconnect();
  }

  // Timer methods
  void stopTimer() {
    timer!.cancel();
    setState(() {
      numH = 0; numM = 0; numS = 0;
      distance = 0;
      duration = 0; totalPwr = 0; totalRPM = 0;
      strH = '00'; strM = '00'; strS = '00';
      isRunning = false;
    });
  }

  void pauseTimer() {
    timer!.cancel();
    setState(() {
      isRunning = false;
    });
  }

  void startTimer() {
    isRunning = true;
    timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      int s = numS + 1, m = numM, h = numH;
      duration += 1;
      // Set timer minutes and hours
      if (s >= 60) {
        if (m >= 60) {
          h += 1; m = 0;
        } else {
          m += 1; s = 0;
        }
      }

      // Update changes to the screen display
      setState(() {
        numS = s; numM = m; numH = h;
        strS = (numS >= 10) ? '$numS' : '0$numS';
        strM = (numM >= 10) ? '$numM' : '0$numM';
        strH = (numH >= 10) ? '$numH' : '0$numH';
        distance += speed;
        totalRPM += rpm;
        totalPwr += power;
      });
    });
  }

  // callback function to set screenId
  setScreenId(val) {
    if (_scaffoldKey.currentState!.isDrawerOpen) {
      _scaffoldKey.currentState!.closeDrawer();
    }
    setState(() {
      screenId = val;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      appBar: AppBar(
        backgroundColor: const Color(0xFF370E4A),
      ),
      drawer: Drawer(
        backgroundColor: const Color(0xFF553265),
        child: Column(
          children: [
            const SizedBox(
              height: 50,
            ),
            Container(
              alignment: Alignment.centerLeft,
              padding: const EdgeInsets.symmetric(horizontal: 10),
              child: Text(
                'Welcome ${FirebaseAuth.instance.currentUser?.displayName}!',
                style: const TextStyle(
                  color: Color(0xFFE3E3E3),
                  fontSize: 16,
                ),
              ),
            ),
            const SizedBox(
              height: 50,
            ),

            SizedBox(
              height: 500,
              child: Column(
                children: [
                  // Home
                  GestureDetector(
                    onTap: () {
                      setScreenId(0);
                    },
                    child: Container(
                      width: 200,
                      alignment: Alignment.center,
                      padding: const EdgeInsets.all(5),
                      decoration: const BoxDecoration(
                        border: Border(
                          bottom: BorderSide(
                            width: 1,
                            color: Color(0xFF9070AA),
                          ),
                        ),
                      ),
                      child: const Text('Home',
                          style: TextStyle(
                            color: Color(0xFFE3E3E3),
                            fontWeight: FontWeight.bold,
                          )),
                    ),
                  ),
                  const SizedBox(
                    height: 20,
                  ),

                  // Edit Profile
                  GestureDetector(
                    onTap: () {
                      setScreenId(1);
                    },
                    child: Container(
                      width: 200,
                      alignment: Alignment.center,
                      padding: const EdgeInsets.all(5),
                      decoration: const BoxDecoration(
                        border: Border(
                          bottom: BorderSide(
                            width: 1,
                            color: Color(0xFF9070AA),
                          ),
                        ),
                      ),
                      child: const Text('Edit Profile',
                          style: TextStyle(
                            color: Color(0xFFE3E3E3),
                            fontWeight: FontWeight.bold,
                          )),
                    ),
                  ),
                  const SizedBox(
                    height: 20,
                  ),

                  // Workout Record
                  GestureDetector(
                    onTap: () async {
                      List snapshotData = await getRecord();
                      setState(() {
                        this.snapshotData = snapshotData.elementAt(0);
                        recordSize = snapshotData.elementAt(1);
                      });
                      setScreenId(2);
                    },
                    child: Container(
                      width: 200,
                      alignment: Alignment.center,
                      padding: const EdgeInsets.all(5),
                      decoration: const BoxDecoration(
                        border: Border(
                          bottom: BorderSide(
                            width: 1,
                            color: Color(0xFF9070AA),
                          ),
                        ),
                      ),
                      child: const Text('Workout Record',
                          style: TextStyle(
                            color: Color(0xFFE3E3E3),
                            fontWeight: FontWeight.bold,
                          )),
                    ),
                  ),
                  const SizedBox(
                    height: 20,
                  ),

                  // Manage Bike
                  GestureDetector(
                    onTap: () {
                      setScreenId(3);
                    },
                    child: Container(
                      width: 200,
                      alignment: Alignment.center,
                      padding: const EdgeInsets.all(5),
                      decoration: const BoxDecoration(
                        border: Border(
                          bottom: BorderSide(
                            width: 1,
                            color: Color(0xFF9070AA),
                          ),
                        ),
                      ),
                      child: const Text('Manage Smart Bike',
                          style: TextStyle(
                            color: Color(0xFFE3E3E3),
                            fontWeight: FontWeight.bold,
                          )),
                    ),
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                ],
              ),
            ),

            // logout Button
            Align(
              alignment: Alignment.bottomCenter,
              child: GestureDetector(
                onTap: () async {
                  if (mqttConnected) {
                    try {
                      await disconnectMqtt();
                    } catch (e) {
                      debugPrint('Error $e');
                    }
                  }
                  await firebaseAuthUtil.logout()
                  .then((e) {
                    Navigator.pushAndRemoveUntil(
                        context,
                        MaterialPageRoute(
                          builder: (context) => Login(),
                        ),
                            (route) => false);
                  });

                  mqttConnected = false;
                },
                behavior: HitTestBehavior.translucent,
                child: Container(
                  alignment: Alignment.center,
                  width: 150,
                  padding: const EdgeInsets.all(5),
                  decoration: BoxDecoration(
                    border:
                        Border.all(color: const Color(0xFFE3E3E3), width: 2),
                    borderRadius: BorderRadius.circular(5),
                  ),
                  child: const Text(
                    'Log out',
                    style: TextStyle(
                      color: Color(0xFFE3E3E3),
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [
              Color(0xFF6E2D51),
              Color(0xFFE97462),
              Color.fromRGBO(55, 14, 74, 0.94),
            ],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            stops: [0.0, 0.5, 1.0],
          ),
        ),
        child: Center(
          child: Container(child: () {
            switch (screenId) {
              case 0:
                return (mqttConnected)
                    ? Column(
                        children: [
                          // Timer Header
                          const SizedBox(
                            height: 50,
                          ),
                          const Text(
                            'Current Session',
                            style: TextStyle(
                              color: Color(0xFFE3E3E3),
                              fontSize: 24,
                            ),
                          ),

                          // Timer display
                          const SizedBox(
                            height: 30,
                          ),
                          Center(
                            child: Text(
                              '$strH:$strM:$strS',
                              style: const TextStyle(
                                color: Color(0xFFE3E3E3),
                                fontSize: 20,
                              ),
                            ),
                          ),

                          // Bike data display
                          const SizedBox(
                            height: 50,
                          ),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.start,
                            children: [
                              Expanded(
                                flex: 2,
                                child: Center(
                                  child: Column(
                                    children: [
                                      const Text(
                                        'Speed',
                                        style: TextStyle(
                                          color: Color(0xFFE3E3E3),
                                          fontSize: 16,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                      const SizedBox(
                                        height: 10,
                                      ),
                                      Text(
                                        '${speed.toStringAsFixed(2)} m/s',
                                        style: const TextStyle(
                                          color: Color(0xFFE3E3E3),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              Expanded(
                                flex: 2,
                                child: Center(
                                  child: Column(
                                    children: [
                                      const Text(
                                        'RPM',
                                        style: TextStyle(
                                          color: Color(0xFFE3E3E3),
                                          fontSize: 16,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                      const SizedBox(
                                        height: 10,
                                      ),
                                      Text(
                                        '${rpm.toStringAsFixed(1)} RPM',
                                        style: const TextStyle(
                                          color: Color(0xFFE3E3E3),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ],
                          ),

                          const SizedBox(
                            height: 50,
                          ),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.start,
                            children: [
                              Expanded(
                                flex: 2,
                                child: Center(
                                  child: Column(
                                    children: [
                                      const Text(
                                        'Power',
                                        style: TextStyle(
                                          color: Color(0xFFE3E3E3),
                                          fontSize: 16,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                      const SizedBox(
                                        height: 10,
                                      ),
                                      Text(
                                        '${power.toStringAsFixed(1)} Watts',
                                        style: const TextStyle(
                                          color: Color(0xFFE3E3E3),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              Expanded(
                                flex: 2,
                                child: Center(
                                  child: Column(
                                    children: [
                                      const Text(
                                        'Distance',
                                        style: TextStyle(
                                          color: Color(0xFFE3E3E3),
                                          fontSize: 16,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                      const SizedBox(
                                        height: 10,
                                      ),
                                      Text(
                                        '${distance.toStringAsFixed(2)} m',
                                        style: const TextStyle(
                                          color: Color(0xFFE3E3E3),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ],
                          ),

                          // temp placeholder for Incline and Resistance setters
                          const SizedBox(
                            height: 50,
                          ),
                          const Center(
                            child: Text(
                              'Resistance',
                              style: TextStyle(
                                color: Color(0xFFE3E3E3),
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              ElevatedButton(
                                onPressed: () {
                                  setState(() {
                                    if (resistance <= 0) return;
                                    resistance -= 0.1;
                                    publishMqtt(
                                        'bike/$bikeId/resistance/control',
                                        resistance);
                                  });
                                },
                                style: ButtonStyle(
                                  backgroundColor:
                                      MaterialStateProperty.all<Color>(
                                    const Color(0xDD553265),
                                  ),
                                ),
                                child: const Text(
                                  '-',
                                  style: TextStyle(
                                    fontSize: 26,
                                  ),
                                ),
                              ),
                              const SizedBox(
                                width: 20,
                              ),
                              Text(
                                resistance.toStringAsFixed(1),
                                style: const TextStyle(
                                  color: Color(0xDDE3E3E3),
                                  fontWeight: FontWeight.bold,
                                  fontSize: 20,
                                ),
                              ),
                              const SizedBox(
                                width: 20,
                              ),
                              ElevatedButton(
                                onPressed: () {
                                  setState(() {
                                    resistance += 0.1;
                                    publishMqtt(
                                        'bike/$bikeId/resistance/control',
                                        resistance);
                                  });
                                },
                                style: ButtonStyle(
                                  backgroundColor:
                                      MaterialStateProperty.all<Color>(
                                    const Color(0xDD553265),
                                  ),
                                ),
                                child: const Text(
                                  '+',
                                  style: TextStyle(
                                    fontSize: 26,
                                  ),
                                ),
                              ),
                            ],
                          ),

                          const SizedBox(
                            height: 30,
                          ),
                          const Center(
                            child: Text(
                              'Incline',
                              style: TextStyle(
                                color: Color(0xFFE3E3E3),
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              ElevatedButton(
                                onPressed: () {
                                  setState(() {
                                    if (incline < -10) return;
                                    incline -= 0.5;
                                    publishMqtt('bike/$bikeId/incline/control',
                                        incline);
                                  });
                                },
                                style: ButtonStyle(
                                  backgroundColor:
                                      MaterialStateProperty.all<Color>(
                                    const Color(0xDD553265),
                                  ),
                                ),
                                child: const Text(
                                  '-',
                                  style: TextStyle(
                                    fontSize: 26,
                                  ),
                                ),
                              ),
                              const SizedBox(
                                width: 20,
                              ),
                              Text(
                                incline.toStringAsFixed(1),
                                style: const TextStyle(
                                  color: Color(0xFFE3E3E3),
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(
                                width: 20,
                              ),
                              ElevatedButton(
                                onPressed: () {
                                  setState(() {
                                    if (incline > 19) return;
                                    incline += 0.5;
                                    publishMqtt('bike/$bikeId/incline/control',
                                        incline);
                                  });
                                },
                                style: ButtonStyle(
                                  backgroundColor:
                                      MaterialStateProperty.all<Color>(
                                    const Color(0xFF553265),
                                  ),
                                ),
                                child: const Text(
                                  '+',
                                  style: TextStyle(
                                    fontSize: 26,
                                  ),
                                ),
                              ),
                            ],
                          ),

                          const SizedBox(
                            height: 20,
                          ),
                          // Start, Pause and Reset Session
                          ElevatedButton(
                              onPressed: () {
                                (!isRunning) ? startTimer() : pauseTimer();
                              },
                              style: ButtonStyle(
                                backgroundColor:
                                    MaterialStateProperty.all<Color>(
                                        (!isRunning)
                                            ? const Color(0xFFA8E898)
                                            : const Color(0xFFDF8D42)),
                              ),
                              child: Text(
                                (!isRunning)
                                    ? 'Start Session'
                                    : 'Pause Session',
                                style: const TextStyle(
                                  color: Color(0xFF370E4A),
                                ),
                              )),

                          ElevatedButton(
                            onPressed: () {
                              addRecord();
                              stopTimer();
                            },
                            style: ButtonStyle(
                                backgroundColor:
                                    MaterialStateProperty.all<Color>(
                                        const Color(0xFFE64036))),
                            child: const Text(
                              'Reset Session',
                              style: TextStyle(
                                color: Color(0xFFE3E3E3),
                              ),
                            ),
                          ),
                        ],
                      )
                    : Column(
                        children: [
                          const Image(
                            image: AssetImage('lib/assets/redbacklogo.png'),
                            height: 150,
                          ),

                          // App Name
                          const Text(
                            "Redback Smart Bike",
                            style: TextStyle(
                              color: Color(0xFFE3E3E3),
                              fontSize: 28,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(
                            height: 50,
                          ),
                          Padding(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 5,
                            ),
                            child: Text(
                              'Welcome ${firebaseAuthUtil.auth.currentUser!.displayName}, thank you for using the Redback Smart Bike mobile application! \nBefore you continue: this application requires a Redback Smart Bike to get started, please connect with your smart bike by entering your bike ID.',
                              textAlign: TextAlign.justify,
                              style: const TextStyle(
                                color: Color(0xFFE3E3E3),
                                fontSize: 16,
                                wordSpacing: 2,
                              ),
                            ),
                          ),

                          const SizedBox(
                            height: 20,
                          ),

                          Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 30),
                            child: TextField(
                              controller: bikeIdController,
                              decoration: InputDecoration(
                                enabledBorder: OutlineInputBorder(
                                  borderSide:
                                      const BorderSide(color: Colors.grey),
                                  borderRadius: BorderRadius.circular(5.0),
                                ),
                                focusedBorder: const OutlineInputBorder(
                                  borderSide: BorderSide(
                                    color: Color(0xFF167EE6),
                                    width: 2.5,
                                  ),
                                ),
                                fillColor: const Color(0xFFD9D9D9),
                                filled: true,
                                hintText: '00000x',
                              ),
                            ),
                          ),

                          const SizedBox(
                            height: 20,
                          ),

                          ElevatedButton(
                            onPressed: () async {
                              //TODO: Validate bikeId with database/server when available
                              List<String> validBikeIdList = [
                                '000001',
                                '000002'
                              ];
                              String bikeIdInput = bikeIdController.text;
                              if (bikeIdInput.isEmpty ||
                                  !validBikeIdList.contains(bikeIdInput)) {
                                Fluttertoast.showToast(
                                    msg: 'Invalid Bike Id',
                                    toastLength: Toast.LENGTH_LONG);
                              } else {
                                bikeId = bikeIdInput;
                                await connectMqtt();
                            }},
                            style: ButtonStyle(
                              backgroundColor: MaterialStateProperty.all<Color>(
                                const Color(0xFF370E4A),
                              ),
                            ),
                            child: const Text('Connect with Smart Bike'),
                          ),
                        ],
                      );
              case 1:
                return EditProfile(callback: setScreenId);
              case 2:
                return WorkoutRecord(
                  listLength: recordSize,
                  snapshot: snapshotData,
                  callback: setScreenId,
                );
              case 3:
                return (bikeId.isNotEmpty)
                    ? Center(
                  child: Column(
                    children: [
                      const SizedBox(height: 20,),
                      Text('Currently connected to Smart Bike #$bikeId.',
                        style: const TextStyle(
                          color: Color(0xFFE3E3E3),
                          fontSize: 14,
                        ),
                      ),
                      const SizedBox(height: 20,),
                      ElevatedButton(
                        onPressed: () {
                          setState(() {
                            disconnectMqtt();
                            setState(() {
                              mqttConnected = false;
                            });
                            bikeId = '';
                            setScreenId(0);
                          });
                        },
                        style: ButtonStyle(
                            backgroundColor: MaterialStateProperty.all<Color>(
                                const Color(0xFF370E4A)
                            )
                        ),
                        child: const Text(
                          'Disconnect Bike',
                          style: TextStyle(
                            color: Color(0xFFE3E3E3),
                            fontSize: 16,
                          ),
                        ),
                      ),
                    ],
                  ),
                )
                    : Center(
                  child: Column(
                    children: [
                      const SizedBox(height: 20,),
                      const Text('Smart bike disconnected, please connect to a valid smart bike.',
                        style: TextStyle(
                          color: Color(0xFFE3E3E3),
                          fontSize: 14,
                        ),
                      ),
                      const SizedBox(height: 20,),
                      ElevatedButton(
                        onPressed: () {
                          setScreenId(0);
                        },
                        style: ButtonStyle(
                            backgroundColor: MaterialStateProperty.all<Color>(
                                const Color(0xFF370E4A)
                            )
                        ),
                        child: const Text(
                          'Connect Smart Bike',
                          style: TextStyle(
                            color: Color(0xFFE3E3E3),
                            fontSize: 16,
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              default:
                return Text('Page No.$screenId Not Implemented!');
            }
          }()),
        ),
      ),
    );
  }
}
