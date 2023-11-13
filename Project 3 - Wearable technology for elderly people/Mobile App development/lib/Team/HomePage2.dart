import 'package:flutter/material.dart';
import 'setting_page.dart';
import 'workout_type.dart';

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _selectedIndex = 0;
  void _onItemTapped(int index)
  {
    setState(() {
      _selectedIndex = index;
    });
  }
  static const List<Widget> _pages =<Widget>[
    HomePage1(),
    Icon(
      Icons.camera,
      size: 150,
    ),
    HomePage1(),
    Icon(
      Icons.chat,
      size: 150,
    ),
    SettingPage(),
  ];

  @override
  Widget build(BuildContext context) {

    return Scaffold(
      body: IndexedStack(
        index: _selectedIndex,
        children: _pages,
      ),
      bottomNavigationBar: BottomNavigationBar(
        elevation: 0,
        iconSize: 40,
        mouseCursor: SystemMouseCursors.grab,
        showSelectedLabels: true,
        showUnselectedLabels: true,
        selectedItemColor: Colors.black54,
        selectedLabelStyle: const TextStyle(fontWeight: FontWeight.bold),
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: 'Home',
            backgroundColor: Color(0xff917A5B),
          ),
          BottomNavigationBarItem(
              icon: Icon(Icons.people),
              label: 'Friends',
              backgroundColor: Color(0xff917A5B)
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.swap_horizontal_circle_sharp, size:50, color: Color(0xff87A395),),
            label: '',
            backgroundColor: Color(0xff917A5B),
          ),
          BottomNavigationBarItem(
              icon: Icon(Icons.star_rate_outlined),
              label: 'Arena',
              backgroundColor: Color(0xff917A5B)
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings_outlined, size:24),
            label: 'Settings',
            backgroundColor: Color(0xff917A5B),
          ),
        ],
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
      ),

      // This trailing comma makes auto-formatting nicer for build methods.
    );
  }
}

class HomePage1 extends StatelessWidget
{
  const HomePage1({super.key});

  @override
  Widget build(BuildContext context) {
    // TODO: implement build
    return DefaultTabController(
        length: 3,
        initialIndex: 0,
        child: Scaffold(
          backgroundColor: const Color(0xff87A395),
          appBar: AppBar(
            backgroundColor: const Color(0xff87A395),
            flexibleSpace: const Column(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                TabBar(
                    tabs: [
                      Tab(text: 'Home',),
                      Tab(text: 'Activities',),
                      Tab(text: 'Profile',),
                    ])
              ],
            ),
          ),
          body: const TabBarView(
            children: [
              HomeTab(),
              Icon(
                Icons.camera,
                size: 150,
              ),
              Icon(
                Icons.chat,
                size: 150,
              ),
            ],
          ),
        )
    );
  }
}
class HomeTab extends StatelessWidget
{
  const HomeTab({super.key});

  @override
  Widget build(BuildContext context)
  {
    return Container(

        padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 30),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 20),
              decoration: BoxDecoration(
                  color: const Color(0xff545C22),
                  borderRadius: BorderRadius.circular(15),
                  boxShadow: const [BoxShadow(color: Colors.black12, spreadRadius: 2)]
              ),
              child:  Column(
                children: <Widget>[
                  Column(
                    children: <Widget>[
                      const Column(
                        children: <Widget>[
                          Text('300.14', style: TextStyle(fontWeight: FontWeight.bold,fontSize: 30,color: Colors.white),),
                          Text('Total Kilometers', style: TextStyle(fontWeight: FontWeight.bold,fontSize: 15,color: Colors.white),),
                        ],
                      ),
                      const SizedBox(height:20),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 30),
                            decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(8),
                                color: const Color(0xff317751)
                            ),
                            child:const Center(
                                child: Column(
                                  children: [
                                    Text('1458.6'),
                                    Text('Avg. KCal'),
                                  ],
                                )
                            ),
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 30),
                            decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(8),
                                color: const Color(0xff317751)
                            ),
                            child:const Center(
                                child: Column(
                                  children: [
                                    Text('1458.6'),
                                    Text('Avg. KCal'),
                                  ],
                                )
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height:20),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 30),
                            decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(8),
                                color: const Color(0xff317751)
                            ),
                            child:const Center(
                                child: Column(
                                  children: [
                                    Text('1458.6'),
                                    Text('Avg. KCal'),
                                  ],
                                )
                            ),
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 30),
                            decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(8),
                                color: const Color(0xff317751)
                            ),
                            child:const Center(
                                child: Column(
                                  children: [
                                    Text('1458.6'),
                                    Text('Avg. KCal'),
                                  ],
                                )
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                  const SizedBox(height:30),
                  const Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      InkWell(
                        child: Text('Leaderboard',
                          style: TextStyle(color:Colors.blue,),
                        ),
                      ),
                      InkWell(
                        child: Text('Stats',
                          style: TextStyle(color:Colors.blue,),
                        ),
                      ),
                      InkWell(
                        child: Text('Challenges',
                          style: TextStyle(color:Colors.blue,),),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height:10),
            Row(
              children: [
                Expanded(child: TextButton(
                  style: TextButton.styleFrom(backgroundColor: const Color(0xff041B09),
                    textStyle: const TextStyle(fontWeight: FontWeight.bold,fontSize: 20,color: Colors.white),
                  ),
                  onPressed: (){
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context)=> const WorkOutTypePage() )
                    );
                  },
                  child: const Text('START', style: TextStyle(fontWeight: FontWeight.bold,fontSize: 20,color: Colors.white),),
                ))
              ],
            )
          ],
        )
    );
  }
}
