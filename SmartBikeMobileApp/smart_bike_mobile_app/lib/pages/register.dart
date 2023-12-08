import 'package:flutter/material.dart';
import 'package:smart_bike_mobile_app/utils/firebase_auth_util.dart';
import 'package:smart_bike_mobile_app/pages/login.dart';

class Register extends StatelessWidget {
  Register({super.key});

  final emailController = TextEditingController();
  final usernameController = TextEditingController();
  final passwordController = TextEditingController();
  final passwordConfirmController = TextEditingController();
  final firebaseAuthUtil = FirebaseAuthUtil();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
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
        child: SafeArea(
          child: Center(
            child: Column(
              children: [
                // Redback Logo
                const Image(
                  image: AssetImage('lib/assets/redbacklogo.png'),
                  height: 150,
                ),

                // App Name
                const Text("Redback Smart Bike",
                  style: TextStyle(
                    color: Color(0xFFE3E3E3),
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                  ),
                ),

                // Email Input
                const SizedBox(
                  height: 50,
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 40.0,
                  ),
                  child: TextField(
                    controller: emailController,
                    decoration: InputDecoration(
                      enabledBorder: OutlineInputBorder(
                        borderSide: const  BorderSide(color: Colors.grey),
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
                      hintText: 'Email',
                    ),
                  ),
                ),

                // Username Input
                const SizedBox(
                  height: 30,
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 40.0,
                  ),
                  child: TextField(
                    controller: usernameController,
                    decoration: InputDecoration(
                      enabledBorder: OutlineInputBorder(
                        borderSide: const  BorderSide(color: Colors.grey),
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
                      hintText: 'Username',
                    ),
                  ),
                ),

                // Password Input
                const SizedBox(
                  height: 30,
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 40.0,
                  ),
                  child: TextField(
                    controller: passwordController,
                    decoration: InputDecoration(
                      enabledBorder: OutlineInputBorder(
                        borderSide: const  BorderSide(color: Colors.grey),
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
                      hintText: 'Password',
                    ),
                    obscureText: true,
                  ),
                ),

                // Confirm Password Input
                const SizedBox(
                  height: 30,
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 40.0,
                  ),
                  child: TextField(
                    controller: passwordConfirmController,
                    decoration: InputDecoration(
                      enabledBorder: OutlineInputBorder(
                        borderSide: const  BorderSide(color: Colors.grey),
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
                      hintText: 'Confirm Password',
                    ),
                    obscureText: true,
                  ),
                ),

                // Register Button
                const SizedBox(
                  height: 50,
                ),
                Container(
                  margin: const EdgeInsets.symmetric(
                    horizontal: 70.0,
                  ),
                  child: FilledButton(
                    onPressed: () {
                      // Local variables to store user input
                      String email = emailController.text;
                      String username = usernameController.text;
                      String password = passwordController.text;
                      String passwordConfirm = passwordConfirmController.text;

                      firebaseAuthUtil.register(email, username, password, passwordConfirm, context);
                    },
                    style: ButtonStyle(
                      backgroundColor: MaterialStateColor.resolveWith((states) => const Color(0xFF370E4A)),
                      shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                        RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(5.0),
                        ),
                      ),
                      minimumSize: MaterialStateProperty.all<Size>(
                        const Size(100, 50),
                      ),
                    ),
                    child: const Center(
                      child: Text("Register",
                        style: TextStyle(
                          color: Color(0xFFE3E3E3),
                          fontWeight: FontWeight.bold,
                          fontSize: 20,
                        ),
                      ),
                    ),
                  ),
                ),

                // Navigation to Register Page
                const SizedBox(
                  height: 30,
                ),
                GestureDetector(
                  onTap: () {
                    Navigator.pushAndRemoveUntil(
                      context,
                      MaterialPageRoute(
                          builder: (context) =>
                          Login(),
                      ), ((Route route) => false)
                    );
                  },
                  child: const Text('Back to Login',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w400,
                      color: Color(0xFF167EE6),
                      decoration: TextDecoration.underline,
                      decorationColor: Color(0xFF167EE6),
                    ),
                  ),
                ),

              ],
            ),
          ),
        ),
      ),
    );
  }
}
