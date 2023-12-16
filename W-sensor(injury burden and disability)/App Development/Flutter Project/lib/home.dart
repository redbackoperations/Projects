import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: HomeScreen1(),
    );
  }
}

class HomeScreen1 extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF302C2C),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            SizedBox(height: 20.0),
            Text(
              'My Data',
              style: TextStyle(
                fontSize: 24.0,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            SizedBox(height: 20.0),
            Container(
              width: 300.0, // Set the width of the larger box
              padding: EdgeInsets.all(20.0),
              decoration: BoxDecoration(
                border: Border.all(
                  width: 2.0,
                  color: Color(0xFF302C2C),
                ),
                color: Colors.blue,
              ),
              child: Column(
                children: <Widget>[
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: <Widget>[
                      DataButton(
                        label: 'Walking',
                        value: '1000',
                      ),
                      DataButton(
                        label: 'Running',
                        value: '700',
                      ),
                    ],
                  ),
                  SizedBox(height: 20.0),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: <Widget>[
                      DataButton(
                        label: 'Jogging',
                        value: '500',
                      ),
                      DataButton(
                        label: 'Cycling',
                        value: '0',
                      ),
                    ],
                  ),
                ],
              ),
            ),
            SizedBox(height: 20.0),
            ElevatedButton(
              onPressed: () {
                // Handle button click here
              },
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Text(
                  'Start',
                  style: TextStyle(color: Colors.white, fontSize: 20.0),
                ),
              ),
              style: ElevatedButton.styleFrom(
                primary: Color(0xFF302C2C),
                minimumSize: Size(200.0, 60.0),
              ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 0, // Set the initial index
        onTap: _onItemTapped,
        type: BottomNavigationBarType.fixed, // Set type to fixed for more than 3 items
        items: <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.data_usage),
            label: 'Data',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.refresh),
            label: 'Refresh',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.people),
            label: 'Friends',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
      ),
    );
  }

  void _onItemTapped(int index) {
    // Print the index of the selected item
    print('Item $index tapped');
  }
}

class DataButton extends StatelessWidget {
  final String label;
  final String value;

  DataButton({
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        // Handle button click here
        print('$label button tapped');
      },
      child: Container(
        width: 100,
        height: 100,
        decoration: BoxDecoration(
          border: Border.all(
            color: Color(0xFF302C2C),
          ),
          color: Colors.blue,
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              label,
              style: TextStyle(
                fontSize: 20.0,
                color: Colors.white,
              ),
            ),
            Text(
              value,
              style: TextStyle(
                fontSize: 24.0,
                color: Colors.white,
              ),
            ),
          ],
        ),
      ),
    );
  }

}
