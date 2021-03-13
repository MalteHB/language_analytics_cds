#!usr/bin/env python3
# Importing packages
import argparse

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from utils.utils import setting_default_data_dir, setting_default_out_dir


def main(args):

    # Importing arguments from the arguments parser

    input_file = args.i

    graph_out_dir = args.god

    save_graph = args.sg

    data_out_dir = args.dad

    min_edge_weight = args.mew

    save_df = args.sdf

    NetworkAnalysis(input_file=input_file,
                    graph_out_dir=graph_out_dir,
                    save_graph=save_graph,
                    data_out_dir=data_out_dir,
                    min_edge_weight=min_edge_weight,
                    save_df=save_df)

    print("DONE! Have a nice day. :-)")


class NetworkAnalysis:
    """Calculates the sentiment scores for a given .csv file.

    Outputs plots of rolling averages across weeks and months.
    """

    def __init__(self,
                 input_file=None,
                 graph_out_dir=None,
                 save_graph=True,
                 data_out_dir=None,
                 min_edge_weight=500,
                 save_df=True):

        # Setting up directories

        self.input_file = input_file

        if self.input_file is None:

            self.input_file = setting_default_data_dir(assigment=4)  # Setting default data directory.

            print(f"\nData directory is not specified.\nSetting it to '{self.input_file}'.")

        self.graph_out_dir = graph_out_dir

        self.save_graph = save_graph

        self.data_out_dir = data_out_dir

        if self.graph_out_dir is None:

            self.graph_out_dir, self.data_out_dir = setting_default_out_dir()  # Setting default output directory.

            print(f"\nOutput directory is not specified.\nSetting it to '{self.graph_out_dir}'.")

            print(f"\nOutput directory is not specified.\nSetting it to '{self.data_out_dir}'.")

        self.graph_out_dir.mkdir(parents=True, exist_ok=True)  # Making sure output directory exists.

        self.data_out_dir.mkdir(parents=True, exist_ok=True)  # Making sure output directory exists.

        self.min_edge_weight = min_edge_weight

        self.save_df = save_df

        # Reading data

        edges_df = pd.read_csv(self.input_file)

        # Creating graph

        graph = self.create_network_graph(edges_df=edges_df,
                                          min_edge_weight=self.min_edge_weight,
                                          save_graph=self.save_graph)

        # Calculating centrality measures

        self.calculate_centrality_measures(graph,
                                           save_df=self.save_df)

    
    def create_network_graph(self, edges_df, min_edge_weight=500, save_graph=True):
        
        # Filter from minimum edge weight

        filtered_edges_df = edges_df[edges_df["weight"] > min_edge_weight]  # Filter from minimum edge weight

        # Create graph

        graph = nx.from_pandas_edgelist(filtered_edges_df, 'nodeA', 'nodeB', ["weight"])        

        pos = nx.spring_layout(graph)
        
        nx.draw(graph, pos, with_labels=True, node_size=10, font_size=10)

        # Save graph

        if save_graph:

            graph_path = self.graph_out_dir / "network_viz.png"

            plt.savefig(graph_path, dpi=300, bbox_inches="tight")

        return graph

    def calculate_centrality_measures(self, graph, save_df=True):

        # Calculate centrality metrics 

        degree = nx.degree_centrality(graph)

        betweenness = nx.betweenness_centrality(graph)

        eigenvector = nx.eigenvector_centrality(graph)

        # Create dataframe

        centrality_df = pd.DataFrame({
            'nodes': list(degree.keys()),
            'degree': list(degree.values()),
            'betweenness': list(betweenness.values()),
            'eigenvector': list(eigenvector.values()),  
        }).sort_values([
            'degree',
            'betweenness',
            'eigenvector'],
            ascending=False)

        # Save dataframe

        if save_df:

            df_path = self.data_out_dir / "centrality_df.csv"

            centrality_df.to_csv(df_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--i',
                        metavar="Input File",
                        type=str,
                        help='A path to the input file.',
                        required=False)

    parser.add_argument('--god',
                        metavar="Graph Output Directory",
                        type=str,
                        help='A path to the graph output directory.',
                        required=False)

    parser.add_argument('--sg',
                        metavar="Save graph",
                        type=bool,
                        help='Whether to save a plot of the graph.',
                        required=False,
                        default=True)

    parser.add_argument('--dad',
                        metavar="Data Output Directory",
                        type=str,
                        help='A path to the data output directory.',
                        required=False)

    parser.add_argument('--mew',
                        metavar="Minimum Edge Weight",
                        type=int,
                        help='The minimum edge weight for the network nodes.',
                        required=False,
                        default=500)

    parser.add_argument('--sdf',
                        metavar="Save Dataframe",
                        type=bool,
                        help='Whether to save the dataframe of centrality measures.',
                        required=False,
                        default=True)

    main(parser.parse_args())
