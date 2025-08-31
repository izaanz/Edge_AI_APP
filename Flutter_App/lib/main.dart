import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'IU Edge AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        scaffoldBackgroundColor: Colors.grey[50],
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.deepPurple,
          foregroundColor: Colors.white,
        ),
      ),
      home: const RecognitionPage(),
    );
  }
}

class RecognitionPage extends StatefulWidget {
  const RecognitionPage({super.key});
  @override
  _RecognitionPageState createState() => _RecognitionPageState();
}

class _RecognitionPageState extends State<RecognitionPage> {
  File? _imageFile;
  bool _isLoading = false;
  Map<String, String> _results = {};
  int _inferenceTime = 0;

  Interpreter? _ageInterpreter;
  Interpreter? _genderInterpreter;
  Interpreter? _emotionInterpreter;

  List<String> _ageLabels = [];
  List<String> _genderLabels = [];
  List<String> _emotionLabels = [];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadModelsAndLabels();
    });
  }
  
  Future<void> _loadModelsAndLabels() async {
    // This function is correct and does not need changes.
    try {
      final interpreterOptions = InterpreterOptions();
      _ageInterpreter = await Interpreter.fromAsset('assets/age_model.tflite', options: interpreterOptions);
      _genderInterpreter = await Interpreter.fromAsset('assets/gender_model.tflite', options: interpreterOptions);
      _emotionInterpreter = await Interpreter.fromAsset('assets/emotion_model.tflite', options: interpreterOptions);
      _ageLabels = await _loadLabels('assets/age_labels.txt');
      _genderLabels = await _loadLabels('assets/gender_labels.txt');
      _emotionLabels = await _loadLabels('assets/emotion_labels.txt');
      print("All models and labels loaded successfully.");
    } catch (e) {
      print("FATAL ERROR loading models or labels: $e");
      setState(() { _results = {'Error': 'Models failed to load.'}; });
    }
  }

  Future<List<String>> _loadLabels(String assetPath) async {
    final labelData = await rootBundle.loadString(assetPath);
    return labelData.split('\n').where((label) => label.isNotEmpty).toList();
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final pickedFile = await ImagePicker().pickImage(source: source);
      if (pickedFile != null) {
        setState(() {
          _imageFile = File(pickedFile.path);
          _isLoading = true;
          _results = {};
        });
        _runInference();
      }
    } catch (e) {
      print("Error picking image: $e");
    }
  }
  
  // ====================================================================
  // THE FIX: TWO SEPARATE PREPROCESSING FUNCTIONS
  // ====================================================================

  /// Prepares image for Age/Gender models (normalize to [0, 1]).
  Float32List _preprocessForAgeGender(img.Image image, int width, int height) {
    final resizedImage = img.copyResize(image, width: width, height: height);
    final inputTensor = Float32List(1 * width * height * 3);
    int pixelIndex = 0;
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        var pixel = resizedImage.getPixel(x, y);
        inputTensor[pixelIndex++] = pixel.r / 255.0;
        inputTensor[pixelIndex++] = pixel.g / 255.0;
        inputTensor[pixelIndex++] = pixel.b / 255.0;
      }
    }
    return inputTensor;
  }

  /// Prepares image for Emotion model (GRAYSCALE and NO normalization [0, 255]).
  Float32List _preprocessForEmotion(img.Image image, int width, int height) {
    final grayscaleImage = img.grayscale(image); // Convert to grayscale first
    final resizedImage = img.copyResize(grayscaleImage, width: width, height: height);
    final inputTensor = Float32List(1 * width * height * 3);
    int pixelIndex = 0;
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        var pixel = resizedImage.getPixel(x, y);
        // The model expects 3 channels, but since it's grayscale, R=G=B.
        // We pass the raw pixel value (0-255) because the model has a Rescaling layer.
        inputTensor[pixelIndex++] = pixel.r.toDouble();
        inputTensor[pixelIndex++] = pixel.g.toDouble();
        inputTensor[pixelIndex++] = pixel.b.toDouble();
      }
    }
    return inputTensor;
  }

  Future<void> _runInference() async {
  if (_imageFile == null || _ageInterpreter == null || _genderInterpreter == null || _emotionInterpreter == null) return;
  
  final stopwatch = Stopwatch()..start();
  try {
    final originalImage = img.decodeImage(await _imageFile!.readAsBytes())!;
    
    // --- CORRECT: Declare as local variables inside the function ---
    String ageResult;
    String genderResult;
    String emotionResult;

    // --- Run Age Model ---
    final ageShape = _ageInterpreter!.getInputTensor(0).shape;
    final ageInput = _preprocessForAgeGender(originalImage, ageShape[1], ageShape[2]);
    final ageOutput = List.filled(1 * 8, 0.0).reshape([1, 8]); // 5 age classes
    _ageInterpreter!.run(ageInput.reshape(ageShape), ageOutput);
    ageResult = _getTopLabel(ageOutput[0], _ageLabels);

    // --- Run Gender Model ---
    final genderShape = _genderInterpreter!.getInputTensor(0).shape;
    final genderInput = _preprocessForAgeGender(originalImage, genderShape[1], genderShape[2]);
    final genderOutput = List.filled(1 * 1, 0.0).reshape([1, 1]); // 1 output value
    _genderInterpreter!.run(genderInput.reshape(genderShape), genderOutput);
    genderResult = _genderLabels[genderOutput[0][0] > 0.5 ? 1 : 0];

    // --- Run Emotion Model ---
    final emotionShape = _emotionInterpreter!.getInputTensor(0).shape;
    final emotionInput = _preprocessForEmotion(originalImage, emotionShape[1], emotionShape[2]);
    final emotionOutput = List.filled(1 * 8, 0.0).reshape([1, 8]); // 8 emotion classes
    _emotionInterpreter!.run(emotionInput.reshape(emotionShape), emotionOutput);
    emotionResult = _getTopLabel(emotionOutput[0], _emotionLabels);
    
    stopwatch.stop();

    setState(() {
      _results = {
        'Age': ageResult,
        'Gender': genderResult,
        'Emotion': emotionResult,
      };
      _isLoading = false;
      _inferenceTime = stopwatch.elapsedMilliseconds;
    });

  } catch (e, s) {
    print("ERROR during inference: $e\n$s");
    setState(() { 
      _isLoading = false; 
      _results = {'Error': 'Inference failed.'}; 
    });
  }
}

  String _getTopLabel(List<dynamic> scores, List<String> labels) {
    double maxScore = 0.0;
    int maxIndex = -1;
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxIndex = i;
      }
    }
    return maxIndex != -1 ? labels[maxIndex] : "Unknown";
  }

  // --- UI Methods (Unchanged) ---
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Age, Gender & Emotion Recognition'), centerTitle: true),
      body: SingleChildScrollView(child: Center(child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const SizedBox(height: 20),
            _buildImageView(),
            const SizedBox(height: 30),
            _buildResultsView(),
            const SizedBox(height: 30),
            _buildButtons(),
            const SizedBox(height: 20),
          ],
        ),
      ))),
    );
  }

  Widget _buildImageView() {
    return Container(
      width: 300, height: 300,
      decoration: BoxDecoration(
        color: Colors.white, borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.deepPurple.withOpacity(0.5), width: 1),
        boxShadow: [BoxShadow(color: Colors.grey.withOpacity(0.2), spreadRadius: 2, blurRadius: 8, offset: const Offset(0, 4))],
      ),
      child: _imageFile == null
          ? const Center(child: Icon(Icons.image_search, size: 60, color: Colors.deepPurple))
          : ClipRRect(borderRadius: BorderRadius.circular(10), child: Image.file(_imageFile!, fit: BoxFit.cover)),
    );
  }

  Widget _buildResultsView() {
    if (_isLoading) return const Column(children: [CircularProgressIndicator(), SizedBox(height: 10), Text("Analyzing...")]);
    if (_results.isEmpty) return const Text('Select an image to analyze', style: TextStyle(fontSize: 18, color: Colors.grey));
    return Card(
      elevation: 5, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            ..._results.entries.map((entry) => _buildResultRow(entry.key, entry.value)).toList(),
            if (_results.isNotEmpty && _results['Error'] == null)
              Padding(
                padding: const EdgeInsets.only(top: 10),
                child: Text('Inference time: $_inferenceTime ms', style: const TextStyle(fontSize: 14, color: Colors.grey)),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultRow(String title, String value) {
    IconData icon;
    Color color = title == 'Error' ? Colors.red : Colors.deepPurple;
    switch (title) {
        case 'Age': icon = Icons.cake_outlined; break;
        case 'Gender': icon = Icons.person_outline; break;
        case 'Emotion': icon = Icons.sentiment_satisfied_outlined; break;
        case 'Error': icon = Icons.error_outline; break;
        default: icon = Icons.label_outline;
    }
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(children: [
            Icon(icon, color: color),
            const SizedBox(width: 16),
            Text('$title:', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: color)),
          ]),
          Text(value, style: TextStyle(fontSize: 18, color: title == 'Error' ? Colors.red : Colors.black87)),
        ],
      ),
    );
  }

  Widget _buildButtons() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        ElevatedButton.icon(
          onPressed: () => _pickImage(ImageSource.camera),
          icon: const Icon(Icons.camera_alt),
          label: const Text('Camera'),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.deepPurple, foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            textStyle: const TextStyle(fontSize: 16),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          ),
        ),
        ElevatedButton.icon(
          onPressed: () => _pickImage(ImageSource.gallery),
          icon: const Icon(Icons.photo_library),
          label: const Text('Gallery'),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.deepPurple, foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            textStyle: const TextStyle(fontSize: 16),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          ),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _ageInterpreter?.close();
    _genderInterpreter?.close();
    _emotionInterpreter?.close();
    super.dispose();
  }
}