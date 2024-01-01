import plotly.graph_objects as go
import plotly.io as pio

class Utils:  

  def plot_table(self, model_data):
    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

    values = model_data
    
    fig = go.Figure(data=[go.Table(
      #columnwidth = [300,300,300,300,300,300],
      header=dict(
        values=['<b>vocab_size</b>','<b>emb_size</b>','<b>number_inputs</b>','<b>total_input_size</b>','<b>num_neurons</b>','<b>trainable_params_size</b>', '<b>train_loss</b>', '<b>dev_loss</b>'],
        line_color='darkslategray',
        fill_color=headerColor,
        align=['left','center'],
        font=dict(color='white', size=12),
      ),
      cells=dict(
        values=values,
        line_color='darkslategray',
        # 2-D list of colors for alternating rows
        fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*5],
        align = ['left', 'center'],
        font = dict(color = 'darkslategray', size = 12)
        ))
    ])

    # fig.show()
    # Save the figure as an image
    #pio.write_image(fig, 'src/neural_language_model/train_perf/training.jpeg')
    fig.write_html("src/neural_language_model/train_perf/training.html")

