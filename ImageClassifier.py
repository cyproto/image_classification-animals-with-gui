import tensorflow as tf, sys
import tkinter as tk
from PIL import Image,ImageTk
import random,os

root = tk.Tk()
root.geometry("1024x768")
root.title("Image Classifier")
bg_img=Image.open("bg.jpeg")
resi = bg_img.resize((1024, 768),Image.ANTIALIAS)
background_image=ImageTk.PhotoImage(resi)
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
background_label.image = background_image

def dispimg():
	imgfile = random.choice(os.listdir("test/"))
	img = Image.open("test/"+imgfile)
	resized = img.resize((500, 400),Image.ANTIALIAS)
	imag = ImageTk.PhotoImage(resized)
	panel = tk.Label(root, image = imag)
	panel.image = imag
	panel.place(x=230,y=130)
	pred(imgfile)
	return img

def pred(imgfile):
	image_path = "/home/cyproto/tf_files/test/"+imgfile

	image_data = tf.gfile.FastGFile(image_path, 'rb').read()

	label_lines = [line.rstrip() for line
                in tf.gfile.GFile("retrained_labels.txt")]

	with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

	
	with tf.Session() as sess:
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		predictions = sess.run(softmax_tensor, \
	 	           {'DecodeJpeg/contents:0': image_data})
	
	arr=[]
	top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
	for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]			
                print('%s (score = %.5f)' % (human_string, score))		
                showtxt(node_id, score)		
	
	
def showtxt(node_id, score):
	
	if node_id == 0 and score >0.40000:
		animal="Monkey"	
	
	elif node_id == 1 and score >0.40000:
		animal="Dog"
		
	elif node_id == 2 and score >0.40000:
		animal="Squirrel"
		
	elif node_id == 3 and score >0.40000:
		animal="Cat"
	
	score0=float(score)
	score0=score0*100	
	text0 = tk.Label(root, text = "    It looks like a "+animal+"     ", font='Arial 30 bold')
	text0.place(x=230,y=540)
	text1 = tk.Label(root, text = "        Accuracy Score :-        ", font='Arial 25 bold')
	text1.place(x=260,y=600)	 
	if node_id == 0:
		text2 = tk.Label(root, text = '   Monkey = %.4f ' %score0, font='Arial 20 bold')
		text2.place(x=320,y=650)		
	elif node_id == 1:
		text3 = tk.Label(root, text = '   Dog = %.4f           ' %score0, font='Arial 20 bold')
		text3.place(x=320,y=650)		
	elif node_id == 2:
		text4 = tk.Label(root, text = '   Squirrel = %.4f ' %score0, font='Arial 20 bold')
		text4.place(x=320,y=650)				
	elif node_id == 3:
		text5 = tk.Label(root, text = '   Cat = %.4f           ' %score0, font='Arial 20 bold')
		text5.place(x=320,y=650)		
	
	text6 = tk.Label(root, text = "  %" , font='Arial 20 bold')
	text6.place(x=525,y=650)

button = tk.Button(root, text='Predict Something', height=100,width=150, bg='WHITE',font='Arial 30 bold', command=dispimg) 
button.config(height=30,width=60)
button.place(x=150,y=30)
button.pack(padx=150,pady=(30,600))
root.mainloop()

