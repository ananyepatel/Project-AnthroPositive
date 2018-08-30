import tornado.httpserver, tornado.ioloop, tornado.options, tornado.web, os.path, random, string
from tornado.options import define, options
from json import dumps
from predict import verify

define("port", default=8888, help="run on the given port", type=int)

image_path = '/home/ananye/PycharmProjects/AnthroPositive/tmp/img.jpg'

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/upload", ImageHandler)
        ]
        tornado.web.Application.__init__(self, handlers)

class ImageHandler(tornado.web.RequestHandler):
    def post(self):
        binaryImageData = self.request.arguments['media'][0]
        
        with open(image_path, 'wb') as writer:
            writer.write(binaryImageData)

        predictor_dict = self.predictor(image_path)
        self.write(dumps(predictor_dict))

    def predictor(self, image_path):
        likelihood, verification = verify(image_path)
        predictDict = dict()

        predictDict['Probability'] = str(likelihood)
        predictDict['Image'] = 'HUMAN' if verification == 0 else 'NOT HUMAN'

        return predictDict

def main():
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()