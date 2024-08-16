import salvo.run
import salvo
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # Create Salvo instance
        salvo = salvo.Salvo()
        # Define arguments
        args = {
            'method': 'GET',
            'concurrency': 4,
            'requests': 1000
        }

        # Log arguments for debugging
        logger.info(f'Arguments: {args}')

        # Run load test
        logger.info('Starting load test...')
        salvo.run.load(url='https://www.youtube.com/', args=args)
        logger.info('Load test completed.')

    except Exception as e:
        # Log any exceptions that occur
        logger.error(f'An error occurred: {e}')


if __name__ == '__main__':
    main()
