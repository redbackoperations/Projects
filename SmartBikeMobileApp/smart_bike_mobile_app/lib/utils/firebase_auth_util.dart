import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:fluttertoast/fluttertoast.dart';

import '../pages/home.dart';
import '../pages/login.dart';

class FirebaseAuthUtil {
  FirebaseAuth auth = FirebaseAuth.instance;

  Future<void> login(String email, String password, BuildContext context) async {
    final auth = FirebaseAuth.instance;

    if (email.isEmpty || password.isEmpty) {
      Fluttertoast.showToast(msg: 'Please enter email and password');
    }
    else {
      try {
        await auth.signInWithEmailAndPassword(email: email, password: password)
            .then((auth) {
          Fluttertoast.showToast(msg: "Login Successful!");
          Navigator.pushAndRemoveUntil(context,
              MaterialPageRoute(builder: (context) =>
              const Home(),
              ), ((Route route) => false));
        });
      } on FirebaseAuthException catch (e) {
        Fluttertoast.showToast(msg: 'Login failed with error: $e');
      }
    }
  }

  Future<void> register(String email, String username, String password, String passwordConfirm, BuildContext context) async {
    final auth = FirebaseAuth.instance;

    // 1. Check if any input is empty
    if (email.isEmpty||password.isEmpty||passwordConfirm.isEmpty) {
      Fluttertoast.cancel();
      Fluttertoast.showToast(
        msg: 'Please fill in all fields in registration form.',
        toastLength: Toast.LENGTH_LONG,
        fontSize: 16,
      );
    }

    // 2. Check if Password matches Confirm Password
    else if (password != passwordConfirm) {
      Fluttertoast.cancel();
      Fluttertoast.showToast(
        msg: 'Passwords do not match!',
        toastLength: Toast.LENGTH_LONG,
        fontSize: 16,
      );
    }

    // 3. Check if password is too short
    else if (password.length < 8) {
      Fluttertoast.cancel();
      Fluttertoast.showToast(
        msg: 'Passwords is too short!',
        toastLength: Toast.LENGTH_LONG,
        fontSize: 16,
      );
    }

    // 4. Firebase authentication
    else {
      try {
        await auth.createUserWithEmailAndPassword(email: email, password: password)
            .then((auth) {
          auth.user!.updateDisplayName(username);
          Fluttertoast.showToast(msg: "User registration completed!");
          Navigator.pushAndRemoveUntil(context,
              MaterialPageRoute(builder: (context) =>
                  Login(),
              ), ((Route route) => false)
          );
        }
        );
      } catch (e) {
        Fluttertoast.showToast(msg: 'Registration Error: $e');
      }
    }
  }

  Future<void> logout() async {
    try {
      await auth.signOut();
    } catch (e) {
      debugPrint('Error: $e');
    }
  }
}