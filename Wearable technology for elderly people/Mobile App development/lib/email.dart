import 'package:flutter/material.dart';

class EmailScreen extends StatefulWidget {
  @override
  _EmailScreenState createState() => _EmailScreenState();
}

class _EmailScreenState extends State<EmailScreen> {
  List<String> registeredEmails = [];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF8F9E91),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
      ),
      body: Container(
        padding: const EdgeInsets.fromLTRB(20, 40, 20, 20),
        decoration: const BoxDecoration(
          color: const Color(0xFF8F9E91),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 20),
            const Center(
              child: DefaultTextStyle(
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 28,
                  color: Colors.white,
                ),
                child: Text('Registered Emails'),
              ),
            ),
            const SizedBox(height: 27),
            Expanded(
              child: ListView.builder(
                itemCount: registeredEmails.length,
                itemBuilder: (context, index) {
                  return ListTile(
                    title: Text(
                      registeredEmails[index],
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                      ),
                    ),
                  );
                },
              ),
            ),
        Align(
          alignment: Alignment.center,
          child: ElevatedButton(
            onPressed: () {
              _showAddEmailDialog(context);
            },
            style: ElevatedButton.styleFrom(
              primary: const Color(0xFF8F9E91),
              minimumSize: Size(200, 50),
            ),
            child: Text(
              'Add Email',
              style: TextStyle(
                fontSize: 18,
                color: Colors.white,
              ),
            ),
          ),
        ),
          ],
        ),
      ),
    );
  }

  void _showAddEmailDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        final emailController = TextEditingController();
        return AlertDialog(
          title: Text('Add Email'),
          content: TextField(
            controller: emailController,
            decoration: InputDecoration(labelText: 'Enter Email'),
          ),
          actions: [
            ElevatedButton(
              onPressed: () {
                final newEmail = emailController.text;
                if (newEmail.isNotEmpty) {
                  setState(() {
                    registeredEmails.add(newEmail);
                  });
                  Navigator.pop(context);
                }
              },
              child: Text('Save Email'),
            ),
            TextButton(
              onPressed: () {
                Navigator.pop(context);
              },
              child: Text('Cancel'),
            ),
          ],
        );
      },
    );
  }
}

void main() {
  runApp(MaterialApp(
    home: EmailScreen(),
  ));
}
