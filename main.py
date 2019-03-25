# -*- coding: utf-8 -*-

from flask import Flask
from flask_restful import Resource, Api,reqparse

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()

class Quotes(Resource):
    def get(self):
        return {
            'pakndul': {
                'quote': ['core of the core.',
                    'ahlinya ahli.',
                    'intinya inti.']
            },
            'linus': {
                'quote': ['Talk is cheap. Show me the code.']
            }

        }

    def post(self):
        parser.add_argument('quote', type=str)
        args = parser.parse_args()

        return {
            'status': True,
            'quote': '{} added. Good'.format(args['quote'])
        }

    def put(self, id):
        parser.add_argument('quote', type=str)
        args = parser.parse_args()

        return {
            'id': id,
            'status': True,
            'quote': 'The quote numbered {} was updated.'.format(id)
        }


#api.add_resource(Quotes, '/')
api.add_resource(Quotes, '/', '/update/<int:id>')




if __name__ == '__main__':
    app.run(debug=True)