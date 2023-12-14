import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:fluttertoast/fluttertoast.dart';


class EditProfile extends StatefulWidget {
  const EditProfile({super.key, required this.callback});
  final Function callback;

  @override
  State<EditProfile> createState() => _EditProfileState();
}

class _EditProfileState extends State<EditProfile> {
  final usernameController = TextEditingController();
  final passwordController = TextEditingController();
  final user = FirebaseAuth.instance.currentUser!;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Center(
        child: Column(
          children: [

            // Email Input
            const SizedBox(
              height: 50,
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
                  hintText: 'New Username',
                ),
              ),
            ),

            // Save Button
            const SizedBox(
              height: 50,
            ),
            Container(
              margin: const EdgeInsets.symmetric(
                horizontal: 70.0,
              ),
              child: FilledButton(
                onPressed: () {
                  updateUserProfile(context, widget.callback, user, usernameController.text);
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
                  child: Text("Save",
                    style: TextStyle(
                      color: Color(0xFFE3E3E3),
                      fontWeight: FontWeight.bold,
                      fontSize: 20,
                    ),
                  ),
                ),
              ),
            ),

            // Alternative Login Text
            const SizedBox(
              height: 30,
            ),
          ],
        ),
      ),
    );
  }
}

Future<void> updateUserProfile(BuildContext context, Function callback, User user, String username) async {
  if (username.isEmpty) {
    Fluttertoast.showToast(msg: 'Please enter a valid username!',
      toastLength: Toast.LENGTH_LONG,
    );
  }
  else {
    try {
      await user.updateDisplayName(username)
          .then((user) {
            callback(0);
      });

    } catch (e) {
      debugPrint('Error in updateUserProfile: $e');
    }
  }
}