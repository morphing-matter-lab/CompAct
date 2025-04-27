from flask import Flask, request, jsonify
from graph import CMGraph
from host import DesignManager
from utils import jsonTypeCheck

manager = DesignManager()
app = Flask(__name__)

@app.route('/')
def default():
    return ''

@app.route('/reset')
def reset():

    manager.reset()
    
    return "reset"

@app.route('/pull', methods=['POST'])
def pull():
    if(not manager.isEmpty):
        graph = manager.getDesign()
        return jsonify(graph.outputJSON()), 200
    else:
        return jsonify({'error': 'Nothing in design history'}), 400

@app.route('/push', methods=['POST'])
def push():

    if request.is_json:
        data = request.get_json()  # Get JSON data from the request
        graph = CMGraph.fromJSON(data)
        if(graph is not None): manager.updateDesign(graph)
        return jsonify(graph.outputJSON()), 200

@app.route('/discretize', methods=['POST'])
def discretize():
    
    if request.is_json:
        data = request.get_json()  # Get JSON data from the request
        graph = CMGraph.initDenseJSON(data)
        if(graph is not None): manager.updateDesign(graph)

        return jsonify(graph.outputJSON()), 200
    else:
        return jsonify({'error': 'Invalid data format'}), 400

@app.route('/topoUpdate', methods=['POST'])
def topoUpdate():
    
    if request.is_json:
        data = request.get_json()  # Get JSON data from the request
        graph = CMGraph.fromJSON(data)
        if(graph is not None): manager.updateDesign(graph)
        
        return jsonify(graph.outputJSON()), 200
    else:
        return jsonify({'error': 'Invalid data format'}), 400

@app.route('/solve', methods=['POST'])
def solve():
    if request.is_json:
        data = request.get_json()  # Get JSON data from the request

        graph = CMGraph.fromJSON(data)
        json = manager.solve(graph, data)
        return jsonify(json), 200
    else:
        return jsonify({'error': 'Invalid data format'}), 400

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.is_json:
        data = request.get_json()  # Get JSON data from the request

        json = manager.analyze(data)
        return jsonify(json), 200
    else:
        return jsonify({'error': 'Invalid data format'}), 400

@app.route('/fix', methods=['POST'])
def fix():
    if request.is_json:
        data = request.get_json()  # Get JSON data from the request

        json = manager.fix(data)
        return jsonify(json), 200
    else:
        return jsonify({'error': 'Invalid data format'}), 400

@app.route('/genModelingInstructions', methods=['POST'])
def genModelingInstructions():
    if(not manager.isEmpty):
        info = manager.genModelingInfo()
        return jsonify(info), 200
    else:
        return jsonify({'error': 'Nothing in design history'}), 400

@app.route('/modeling', methods=['POST'])
def modelingUpdate():

    if request.is_json:
        data = request.get_json()  # Get JSON data from the request

        json = manager.modelingUpdate(data)
        return jsonify(json), 200
    else:
        return jsonify({'error': 'Invalid data format'}), 400

@app.route('/undo', methods=['POST'])
def undo():

    if(not manager.isEmpty):
        graph = manager.undo()
        return jsonify(graph.outputJSON()), 200
    else:
        return jsonify({'error': 'Nothing in design history'}), 400

@app.route('/redo', methods=['POST'])
def redo():

    if(not manager.isEmpty):
        graph = manager.redo()
        return jsonify(graph.outputJSON()), 200
    else:
        return jsonify({'error': 'Nothing in design history'}), 400

if __name__ == '__main__':
    app.run(port=5000)