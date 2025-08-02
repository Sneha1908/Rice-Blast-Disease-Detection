import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(RiceBlastApp());
}

class RiceBlastApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Rice Blast Detection',
      theme: ThemeData(primarySwatch: Colors.green),
      home: DetectionPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class DetectionPage extends StatefulWidget {
  @override
  _DetectionPageState createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  File? _image;
  String _label = '';
  String _confidence = '';
  String _stage = '';
  String _recommendation = '';
  String _outputImageUrl = '';
  String _error = '';
  bool _isLoading = false;

  final String backendURL = 'http://10.232.223.202:5000/detect';

  Future<void> pickImage() async {
    final pickedFile =
    await ImagePicker().pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _label = '';
        _confidence = '';
        _stage = '';
        _recommendation = '';
        _outputImageUrl = '';
        _error = '';
      });
      await uploadAndDetect(File(pickedFile.path));
    }
  }

  Future<void> uploadAndDetect(File imageFile) async {
    setState(() {
      _isLoading = true;
    });

    var request = http.MultipartRequest('POST', Uri.parse(backendURL));
    request.files
        .add(await http.MultipartFile.fromPath('image', imageFile.path));

    try {
      var response = await request.send();
      final res = await http.Response.fromStream(response);

      if (res.statusCode == 200) {
        final data = json.decode(res.body);

        setState(() {
          _label = data['label'];
          _confidence = data['confidence'];
          _isLoading = false;

          if (data.containsKey('message')) {
            _error = data['message'];
          } else {
            _stage = data['detections'][0]['stage'];
            _recommendation = data['detections'][0]['recommendation'];
            _outputImageUrl = 'http://10.232.223.202:5000/' + data['output_image'];

          }
        });
      } else {
        setState(() {
          _isLoading = false;
          _error = 'Server error: ${res.statusCode}';
        });
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _error = 'Error: $e';
      });
    }
  }

  Widget resultCard(String title, String value) {
    return Card(
      margin: EdgeInsets.symmetric(vertical: 4, horizontal: 12),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      elevation: 3,
      child: ListTile(
        title: Text(title, style: TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text(value, style: TextStyle(fontSize: 16)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Rice Blast Detection'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          children: [
            _image != null
                ? Image.file(_image!, height: 200)
                : Container(
              height: 200,
              color: Colors.grey[300],
              child: Center(child: Text("No image selected")),
            ),
            SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: pickImage,
              icon: Icon(Icons.upload_file),
              label: Text('Select Image'),
              style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.symmetric(horizontal: 24, vertical: 12)),
            ),
            SizedBox(height: 20),
            if (_isLoading) CircularProgressIndicator(),

            if (_error.isNotEmpty) resultCard("Error", _error),
            if (_label.isNotEmpty) resultCard("Classification", _label),
            if (_confidence.isNotEmpty) resultCard("Confidence", _confidence),
            if (_stage.isNotEmpty) resultCard("Disease Stage", _stage),
            if (_recommendation.isNotEmpty)
              resultCard("Recommendation", _recommendation),

            if (_outputImageUrl.isNotEmpty)
              Column(
                children: [
                  SizedBox(height: 20),
                  Text("Detection Output",
                      style: TextStyle(fontWeight: FontWeight.bold)),
                  SizedBox(height: 10),
                  Image.network(_outputImageUrl, height: 250),
                ],
              ),
          ],
        ),
      ),
    );
  }
}
