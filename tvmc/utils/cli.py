import click

@click.group()
@click.option('--L', type=int, required=False, help='Total lattice size (8x8 would be L=64).')
@click.option('--Q', type=int, required=False, help='Number of minibatches per batch.')
@click.option('--K', type=int, required=False, help='Size of each minibatch.')
@click.option('--B', type=int, required=False, help='Total batch size (should be Q*K).')
@click.option('--NLOOPS', type=int, required=False, help='Number of loops within the off_diag_labels function.')
@click.option('--steps', type=int, required=False, help='Number of training steps.')
@click.option('--dir', type=str, required=False, help='Output directory, set to <NONE> for no output.')
@click.option('--lr', type=float, required=False, help='Learning rate.')
@click.option('--seed', type=int, required=False, help='Random seed for the run.')
@click.option('--sgrad', is_flag=True, help='Whether or not to sample with gradients.')
@click.option('--true_grad', is_flag=True, help='Set to false to approximate the gradients.')
@click.option('--sub_directory', type=str, default='', help='String to add to the end of the output directory.')
def cli(L, Q, K, B, NLOOPS, steps, dir, lr, seed, sgrad, true_grad, sub_directory):
    """Training CLI for different models."""
    if B != Q * K:
        raise ValueError("Total batch size (B) must be equal to Q * K")
    click.echo("Base Training Configuration:")
    click.echo(f"Lattice size (L): {L}")
    click.echo(f"Minibatches per batch (Q): {Q}")
    click.echo(f"Size of each minibatch (K): {K}")
    click.echo(f"Total batch size (B): {B}")
    click.echo(f"Number of loops (NLOOPS): {NLOOPS}")
    click.echo(f"Training steps: {steps}")
    click.echo(f"Output directory: {dir}")
    click.echo(f"Learning rate: {lr}")
    click.echo(f"Random seed: {seed}")
    click.echo(f"Sample with gradients (sgrad): {sgrad}")
    click.echo(f"Use true gradients (true_grad): {true_grad}")
    click.echo(f"Sub-directory: {sub_directory}")

@click.command()
@click.option('--Nh', type=int, required=False, help='RNN hidden size.')
@click.option('--patch', type=str, required=False, help='Number of atoms input/predicted at once (patch size).')
@click.option('--rnntype', type=click.Choice(['ELMAN', 'GRU']), required=False, help='Type of RNN cell to use.')
def rnn(Nh, patch, rnntype):
    """Configure RNN parameters."""
    click.echo(f"RNN Configuration: Nh={Nh}, patch={patch}, rnntype={rnntype}")

@click.command()
@click.option('--Nh', type=int, required=False, help='Transformer token size.')
@click.option('--patch', type=str, required=False, help='Patch size.')
@click.option('--dropout', type=float, required=False, help='Dropout rate.')
@click.option('--num_layers', type=int, required=False, help='Number of transformer layers.')
@click.option('--nhead', type=int, required=False, help='Number of attention heads.')
@click.option('--repeat_pre', is_flag=True, help='Repeat the precondition instead of projecting.')
def ptf(Nh, patch, dropout, num_layers, nhead, repeat_pre):
    """Configure Patched Transformer (PTF) parameters."""
    click.echo(f"PTF Configuration: Nh={Nh}, patch={patch}, dropout={dropout}, num_layers={num_layers}, nhead={nhead}, repeat_pre={repeat_pre}")

@click.command()
@click.option('--Nh', type=int, required=False, help='Transformer token size.')
@click.option('--patch', type=int, required=False, help='Patch size.')
@click.option('--dropout', type=float, required=False, help='Dropout rate.')
@click.option('--num_layers', type=int, required=False, help='Number of transformer layers.')
@click.option('--nhead', type=int, required=False, help='Number of attention heads.')
@click.option('--subsampler', type=str, required=False, help='Inner model used for probability factorization.')
def lptf(Nh, patch, dropout, num_layers, nhead, subsampler):
    """Configure Large-Patched Transformer (LPTF) parameters."""
    click.echo(f"LPTF Configuration: Nh={Nh}, patch={patch}, dropout={dropout}, num_layers={num_layers}, nhead={nhead}, subsampler={subsampler}")

cli.add_command(rnn)
cli.add_command(ptf)
cli.add_command(lptf)

if __name__ == '__main__':
    cli()